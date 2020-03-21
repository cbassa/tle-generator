#!/usr/bin/env python3
import sys
import argparse
import configparser
import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from tlegenerator.formats import decode_iod_observation
from tlegenerator.observation import read_observers, Dataset
from tlegenerator.twoline import TwoLineElement, format_tle, propagate
from tlegenerator.twoline import read_tles_from_file, find_tle_before
from tlegenerator.optimize import residuals, chisq, rms, track_residuals
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Preprocess observations.")
    parser.add_argument("-c", "--catalog", type=str,
                        help="Input TLE catalog to update",
                        metavar="FILE")
    parser.add_argument("-d", "--data", type=str,
                        help="File with observations")
    parser.add_argument("-i", "--ident", type=int,
                        help="NORAD ID to update")
    parser.add_argument("-t", "--endtime",
                        help="Use observations upto this time (YYYY-MM-DDTHH:MM:SS) [default: now]")
    parser.add_argument("-m", "--endmjd",
                        help="Use observations upto this time (MJD; Modified Julian Day) [default: now]")
    parser.add_argument("-l", "--length", type=float,
                        help="Timespan (in days) to use for fitting) [default: 30]",
                        default=30.0)
    parser.add_argument("-C", "--conf_file",
                        help="Specify configuration file. If no file" +
                        " is specified 'configuration.ini' is used.",
                        metavar="FILE")
    args = parser.parse_args()

    # Read configuration file
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    conf_file = args.conf_file if args.conf_file else "configuration.ini"
    cfg.read(conf_file)

    # Set up logging
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] " +
                                     "[%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    # Attach handler
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    # Read observers
    logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
    observers = read_observers(cfg.get("Common", "observers_file"))

    # Read observations
    try:
        logger.info(f"Reading observations from {args.data}")
        with open(args.data, errors="replace") as f:
            newlines = f.readlines()

            # Parse observations
            observations = [decode_iod_observation(newline, observers) for newline in newlines]
            logger.info(f"{len(observations)} observations read")
    except IOError as e:
        logger.info(e)
        sys.exit()

    # Read TLE catalog
    logger.info(f"Reading TLEs from {args.catalog}")
    tles = read_tles_from_file(args.catalog)
    if tles is None:
        logger.info(f"Error reading {args.catalog}")
        sys.exit()
    logger.info(f"{len(tles)} TLEs read")

    # Restructure observations
    logger.info("Converting observations")
    d = Dataset(observations)

    # Select observations
    if (args.endtime is None) and (args.endmjd is None):
        # Use latest observation
        tend = np.max(d.tobs)
    elif (args.endtime is None) and (args.endmjd is not None):
        tend = Time(args.endmjd, format="mjd")
    else:
        try:
            tend = Time(args.endtime, format="isot", scale="utc")
        except:
            logging.info(f"Failed to parse {args.endtime}")
            sys.exit
    logging.info(f"Selecting observations before {tend.isot}")
    c = d.tobs < tend
    if np.sum(c) == 0:
        logging.info(f"No observations selected")
        sys.exit()
        
    tmax = np.max(d.tobs[c])
    tmin = tmax - args.length * u.d
    d.mask = (d.tobs > tmin) & (d.tobs <= tmax)
    logger.info(f"Last observation obtained at {tmax.isot}")
    logger.info(f"{np.sum(d.mask)} observations selected")

    # Select newest TLE before epoch
    tle = find_tle_before(tles, args.ident, tmax)
    tleage = tmax - Time(tle.epoch, scale="utc")
    if tleage > 0.01:
        logger.info(f"Latest tle ({tle.epochyr:02d}{tle.epochdoy:012.8f}) is {tleage.to(u.d).value:.2f} days old")
    else:
        sys.exit()
    
    # Propagate
    propepoch = tmax.datetime
    proptle, converged = propagate(tle, propepoch, drmin=1e-3, dvmin=1e-6, niter=100)
    logger.info(f"Propagating TLE to {proptle.epochyr:02d}{proptle.epochdoy:012.8f}")

    # Extract parameters
    p = np.array([proptle.incl, proptle.node, proptle.ecc, proptle.argp, proptle.m, proptle.n, proptle.bstar])

    # Compute prefit RMS
    prefit_rms = rms(residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))
    
    # Optimize
    logger.info("Optimize least-squares fit")
    for i in range(10):
        p = optimize.fmin(chisq, p, args=(proptle.satno, proptle.epochyr, proptle.epochdoy, d), disp=False)

    postfit_rms = rms(residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))
    logger.info(f"Pre-fit residuals {prefit_rms:.4f} degrees")
    logger.info(f"Post-fit residuals {postfit_rms:.4f} degrees")

    # Format TLE
    line0, line1, line2 = format_tle(proptle.satno, proptle.epochyr, proptle.epochdoy, *p, proptle.name, proptle.desig)
    newtle = TwoLineElement(line0, line1, line2)
    
    logger.info(f"{line0}")
    logger.info(f"{line1}")
    logger.info(f"{line2}")

    # Get in-track, cross-track residuals
    dt, dr = track_residuals(newtle, d)

    # Apply selection
    tobs = d.tobs[d.mask]
    terr = d.terr[d.mask]
    perr = d.perr[d.mask]
    sites = d.site_id[d.mask]

    # Time string for storage
    tstr = Time(newtle.epoch, format="datetime", scale="utc").isot.replace("-", "").replace(":", "").replace("T", "_")[:15]
    
    # Generate figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

    uniq_sites = np.unique(sites)
    sequence = np.arange(len(dt))
    for site in uniq_sites:
        c = site == sites
        ax1.errorbar(tobs[c].datetime, dt[c], yerr=terr[c], fmt=".", label=f"{site:d}")
        ax2.errorbar(tobs[c].datetime, dr[c], yerr=perr[c], fmt=".", label=f"{site:d}")
        ax3.errorbar(sequence[c], dt[c], yerr=terr[c], fmt=".", label=f"{site:d}")
        ax4.errorbar(sequence[c], dr[c], yerr=perr[c], fmt=".", label=f"{site:d}")

    ax1.set_title(f"{newtle.name} [{newtle.satno}/{newtle.desig}]: {newtle.epochyr:02d}{newtle.epochdoy:012.8f}, {np.sum(d.mask)} measurements, {rms(dt):.4f} sec, {rms(dr):.4f} deg rms")
        
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(7))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator(7))
        
    dtmax, drmax = np.max(np.abs(dt)), np.max(np.abs(dr))
    ax1.set_ylim(-1.5 * dtmax, 1.5 * dtmax)
    ax2.set_ylim(-1.5 * drmax, 1.5 * drmax)
    ax3.set_ylim(-1.5 * dtmax, 1.5 * dtmax)
    ax4.set_ylim(-1.5 * drmax, 1.5 * drmax)
    ax1.axhline(0, color="k")
    ax2.axhline(0, color="k")
    ax3.axhline(0, color="k")
    ax4.axhline(0, color="k")
    ax3.set_xlabel("Sequence")
    ax4.set_xlabel("Sequence")

    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(6))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(6))
    ax4.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.set_ylabel("Time offset (s)")
    ax2.set_ylabel(r"Angular offset ($^\circ$)")
    ax3.set_ylabel("Time offset (s)")
    ax4.set_ylabel(r"Angular offset ($^\circ$)")
    ax1.legend(ncol=len(uniq_sites))
    plt.tight_layout()
    plt.savefig(f"{newtle.satno:05d}_{tstr}_postfit.png", bbox_inches="tight")
 
    # Store
    with open(args.catalog, "a+") as f:
        f.write(f"{line0}\n{line1}\n{line2}\n")
        f.write(f"# {tmin.mjd}-{tmax.mjd}, {np.sum(d.mask)} measurements, {postfit_rms:.4f} deg rms\n")

