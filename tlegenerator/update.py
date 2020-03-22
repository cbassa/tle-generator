#!/usr/bin/env python3
import os
import logging
import numpy as np
from scipy import optimize
import astropy.units as u
from astropy.time import Time
from tlegenerator.observation import Dataset
from tlegenerator.formats import decode_iod_observation
from tlegenerator.twoline import read_tles_from_file, find_tle_before
from tlegenerator.twoline import TwoLineElement, format_tle, propagate
from tlegenerator.optimize import residuals, chisq, rms, track_residuals
from tlegenerator.optimize import format_time_for_output
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

def update_tle(satno, tlefile, datfile, observers, tend, length):
    # Read observations
    observations = None
    if os.path.exists(datfile):
        logging.info(f"Reading observations from {datfile}")
        with open(datfile, errors="replace") as f:
            newlines = f.readlines()
            
            # Parse observations
            observations = [decode_iod_observation(newline, observers) for newline in newlines]
            logging.info(f"{len(observations)} observations read for object {satno}")

    # Read TLE
    logging.info(f"Reading TLEs from {tlefile}")
    tles = read_tles_from_file(tlefile)
    if tles is None:
        logging.info(f"Error reading {tlefile}")
        return
    logging.info(f"{len(tles)} TLEs read")

    # Restructure observations
    logging.info("Converting observations")
    d = Dataset(observations)

    # Find last observation
    if tend is None:
        tend = np.max(d.tobs)
    logging.info(f"Selecting observations before {tend.isot}")
    c = d.tobs <= tend
    if np.sum(c) == 0:
        logging.info(f"No observations selected")
        return

    # Select observations
    tmax = np.max(d.tobs[c])
    tmin = tmax - length * u.d
    d.mask = (d.tobs > tmin) & (d.tobs <= tmax)
    logging.info(f"Last observation obtained at {tmax.isot}")
    logging.info(f"{np.sum(d.mask)} observations selected")

    # Select newest TLE before epoch
    tle = find_tle_before(tles, satno, tmax)
    tleage = tmax - Time(tle.epoch, scale="utc")
    logging.info(f"Latest tle ({tle.epochyr:02d}{tle.epochdoy:012.8f}) is {tleage.to(u.d).value:.4f} days old")
    if tleage <= 0.01:
        logging.info("No need to update TLE")
        return

    # Propagate
    propepoch = tmax.datetime
    proptle, converged = propagate(tle, propepoch, drmin=1e-3, dvmin=1e-6, niter=100)
    logging.info(f"Propagating TLE to {proptle.epochyr:02d}{proptle.epochdoy:012.8f}")

    # Extract parameters
    p = np.array([proptle.incl, proptle.node, proptle.ecc, proptle.argp, proptle.m, proptle.n, proptle.bstar])

    # Compute prefit RMS
    prefit_rms = rms(residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))
    
    # Optimize
    logging.info("Optimize least-squares fit")
    for i in range(10):
        p = optimize.fmin(chisq, p, args=(proptle.satno, proptle.epochyr, proptle.epochdoy, d), disp=False)

    # Compute postfit RMS
    postfit_rms = rms(residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))
    logging.info(f"Pre-fit residuals {prefit_rms:.4f} degrees")
    logging.info(f"Post-fit residuals {postfit_rms:.4f} degrees")

    # Format TLE
    line0, line1, line2 = format_tle(proptle.satno, proptle.epochyr, proptle.epochdoy, *p, proptle.name, proptle.desig)
    newtle = TwoLineElement(line0, line1, line2)
    
    logging.info(f"{line0}")
    logging.info(f"{line1}")
    logging.info(f"{line2}")
    
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

    ax1.set_title(f"{newtle.name} [{newtle.satno}/{newtle.desig}]: {newtle.epochyr:02d}{newtle.epochdoy:012.8f}, {np.sum(d.mask)} measurements, {rms(dt):.4f} sec, {rms(dr):.4f} deg rms", loc="left")
        
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(7))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator(7))
        
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
    ax1.legend(ncol=len(uniq_sites), loc="upper center")
    plt.tight_layout()
    plt.savefig(f"{newtle.satno:05d}_{tstr}_postfit.png", bbox_inches="tight")
    plt.close()
        
    # Store
    with open(tlefile, "a+") as f:
        f.write(f"{line0}\n{line1}\n{line2}\n")
        f.write(f"# {format_time_for_output(tmin)}-{format_time_for_output(tmax)}, {np.sum(d.mask)} measurements, {postfit_rms:.4f} deg rms\n")

    with open("latest.txt", "a+") as f:
        f.write(f"{line0}\n{line1}\n{line2}\n")
        f.write(f"# {format_time_for_output(tmin)}-{format_time_for_output(tmax)}, {np.sum(d.mask)} measurements, {postfit_rms:.4f} deg rms\n")
            
