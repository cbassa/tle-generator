#!/usr/bin/env python3
import os
import sys
import argparse
import configparser
import logging
import yaml

import numpy as np
import astropy.units as u
from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

from tlegenerator import observation as obs
from tlegenerator import formats as fmt
from tlegenerator import database as db
from tlegenerator import twoline
from tlegenerator import optimize
from tlegenerator import update

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Compute residuals.")
    parser.add_argument("-y", "--yaml", type=str,
                        help="Input YAML file with TLE and observations",
                        metavar="FILE")
    parser.add_argument("-C", "--conf_file",
                        help="Specify configuration file. [default: configuration.ini]",
                        metavar="FILE", default="configuration.ini")
    args = parser.parse_args()

    # Set up logging
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] " +
                                     "[%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    # Attach handler
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)

    # Disable matplot logging
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    # Input checking
    if not os.path.exists(args.conf_file):
        logger.error(f"{args.conf_file} not found")
        sys.exit()
        
    # Read configuration file
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(args.conf_file)

    # Read observers
    logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
    observers = obs.read_observers(cfg.get("Common", "observers_file"))

    # Read yaml
    with open(args.yaml, "r") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    # Extract TLE
    tle = twoline.TwoLineElement(*list(data["prefit"].values()))

    # Extract observations
    observations = [fmt.decode_iod_observation(line, observers) for line in data["observations"]]

    # Generate dataset
    d = obs.Dataset(observations)

    # New epoch time
    tmax = np.max(d.tobs)
    tmin = np.min(d.tobs)

    # Propagate
    #propepoch = tmax.datetime
    #proptle, converged = twoline.propagate(tle, propepoch, drmin=1e-3, dvmin=1e-6, niter=100)

    # Extract parameters
    p = np.array([tle.incl, tle.node, tle.ecc, tle.argp, tle.m, tle.n, tle.bstar])

    # Compute prefit RMS
    prefit_rms = optimize.rms(optimize.residuals(p, tle.satno, tle.epochyr, tle.epochdoy, d))
    
    # Get in-track, cross-track residuals
    dt, dr = optimize.track_residuals(tle, d)

    # Apply selection
    tobs = d.tobs[d.mask]
    terr = d.terr[d.mask]
    perr = d.perr[d.mask]
    sites = d.site_id[d.mask]

    # Time string for storage
    tstr = Time(tle.epoch, format="datetime", scale="utc").isot.replace("-", "").replace(":", "").replace("T", "_")[:15]
    
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

    #ax1.set_title(f"{tle.name} [{tle.satno}/{tle.desig}]: {tle.epochyr:02d}{tle.epochdoy:012.8f}, {np.sum(d.mask)} measurements, {rms(dt):.4f} sec, {rms(dr):.4f} deg rms", loc="left")
    ax1.set_title(f"{tle.line0}\n{tle.line1}\n{tle.line2}\n# {optimize.format_time_for_output(tmin)}-{optimize.format_time_for_output(tmax)}, {np.sum(d.mask)} obs, {optimize.rms(dt):.4f} sec, {optimize.rms(dr):.4f} deg rms\n", loc="left", family="monospace")

    print(prefit_rms)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.yaxis.set_minor_locator(AutoMinorLocator())
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    
    dtmax, drmax = np.max(np.abs(dt)), np.max(np.abs(dr))
    ax1.set_ylim(-1.5 * dtmax, 3 * dtmax)
    ax2.set_ylim(-1.5 * drmax, 1.5 * drmax)
    ax3.set_ylim(-1.5 * dtmax, 1.5 * dtmax)
    ax4.set_ylim(-1.5 * drmax, 1.5 * drmax)
    ax1.axhline(0, color="k")
    ax2.axhline(0, color="k")
    ax3.axhline(0, color="k")
    ax4.axhline(0, color="k")
    ax3.set_xlabel("Sequence")
    ax4.set_xlabel("Sequence")
        
    tmin = tmin - 1 * u.d
    tmax = tmax + 1 * u.d
    ax1.set_xlim(tmin.datetime, tmax.datetime)
    ax2.set_xlim(tmin.datetime, tmax.datetime)
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.set_ylabel("Time offset (s)")
    ax2.set_ylabel(r"Angular offset ($^\circ$)")
    ax3.set_ylabel("Time offset (s)")
    ax4.set_ylabel(r"Angular offset ($^\circ$)")
    ax1.legend(ncol=len(uniq_sites), loc="upper center")
    plt.tight_layout()
    #plt.savefig(f"results/{tle.satno:05d}_{tstr}_postfit.png", bbox_inches="tight")
    plt.show()
    plt.close()
