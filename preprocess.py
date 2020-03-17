#!/usr/bin/env python3
import sys
import argparse
import configparser
import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from datetime import datetime
from tlegenerator.iod import decode_iod_observation
from tlegenerator.observation import read_observers, Dataset
from tlegenerator.twoline import TwoLineElement, format_tle, propagate
from sgp4.api import Satrec
from scipy import optimize

def residuals(a, satno, epochyr, epochdoy, d):
    # Format TLE from parameters
    line0, line1, line2 = format_tle(satno, epochyr, epochdoy, *a)

    # Set up satellite
    sat = Satrec.twoline2rv(line1, line2)
    
    # Compute integer and fractional JD
    jdint = np.floor(d.tobs.jd[d.mask])
    jdfrac = d.tobs.jd[d.mask] - jdint
    
    # Evaluate SGP4
    e, rsat_teme, vsat_teme = sat.sgp4_array(jdint, jdfrac)

    # Convert to GCRS
    rsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], rsat_teme)
    vsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], vsat_teme)

    # Compute unit vectors
    dr = rsat - d.robs[d.mask]
    r = np.linalg.norm(dr, axis=1)
    upred = dr / r[:, np.newaxis]

    # Compute residuals (in degrees)
    res = np.arccos(np.sum(d.uobs[d.mask] * upred, axis=1)) * 180 / np.pi
    
    return res

def chisq(a, satno, epochyr, epochdoy, d):
    return np.sum(residuals(a, satno, epochyr, epochdoy, d)**2)

def rms(x):
    return np.sqrt(np.sum(x**2) / len(x))


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
    try:
        logger.info(f"Reading TLEs from {args.catalog}")
        with open(args.catalog, "r") as f:
            lines = f.readlines()

            # Parse TLEs
            tles = [TwoLineElement(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines), 3)]
            logger.info(f"{len(tles)} TLEs read")
    except IOError as e:
        logger.info(e)
        sys.exit()

    # Restructure observations
    d = Dataset(observations)

    # Select observations
    tmax = np.max(d.tobs)
    tmin = tmax - args.length * u.d
    d.mask = (d.tobs >= tmin) & (d.tobs < tmax)
    logger.info(f"Last observation obtained at {tmax.isot}")
    logger.info(f"{np.sum(d.mask)} observations selected")

    # TODOD: Add automatic selection of closest TLE
    tle = tles[0]
    tleage = tmax - Time(tle.epoch, scale="utc")
    logger.info(f"Latest tle ({tle.epochyr:02d}{tle.epochdoy:012.8f}) is {tleage.to(u.d).value:.2f} days old")
    
    # Propagate
    newepoch = tmax.datetime
    newtle, converged = propagate(tle, newepoch)
    logger.info(f"Propagating TLE to {newtle.epochyr:02d}{newtle.epochdoy:012.8f}")
    
    # Extract parameters
    p = np.array([newtle.incl, newtle.node, newtle.ecc, newtle.argp, newtle.m, newtle.n, newtle.bstar])

    # Compute prefit RMS
    prefit_rms = rms(residuals(p, newtle.satno, newtle.epochyr, newtle.epochdoy, d))
    
    # Optimize
    logger.info("Optimize least-squares fit")
    for i in range(10):
        p = optimize.fmin(chisq, p, args=(newtle.satno, newtle.epochyr, newtle.epochdoy, d), disp=False)

    postfit_rms = rms(residuals(p, newtle.satno, newtle.epochyr, newtle.epochdoy, d))
    logger.info(f"Pre-fit residuals {prefit_rms:.4f} degrees")
    logger.info(f"Post-fit residuals {postfit_rms:.4f} degrees")
    
    line0, line1, line2 = format_tle(newtle.satno, newtle.epochyr, newtle.epochdoy, *p, newtle.name, newtle.desig)
    print(f"{line0}\n{line1}\n{line2}")

