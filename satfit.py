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
from scipy import optimize as scipy_optimize

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
    propepoch = tmax.datetime
    proptle, converged = twoline.propagate(tle, propepoch, drmin=1e-3, dvmin=1e-6, niter=100)

    # Extract parameters
    p = np.array([proptle.incl, proptle.node, proptle.ecc, proptle.argp, proptle.m, proptle.n, proptle.bstar])

    # Compute prefit RMS
    prefit_rms = optimize.rms(optimize.residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))

    # Optimize
    for i in range(10):
        p = scipy_optimize.fmin(optimize.chisq, p, args=(proptle.satno, proptle.epochyr, proptle.epochdoy, d), disp=False)

    # Compute postfit RMS
    postfit_rms = optimize.rms(optimize.residuals(p, proptle.satno, proptle.epochyr, proptle.epochdoy, d))

   # Format TLE
    line0, line1, line2 = twoline.format_tle(proptle.satno, proptle.epochyr, proptle.epochdoy, *p, proptle.name, proptle.desig)
    newtle = twoline.TwoLineElement(line0, line1, line2)
    
    # Get in-track, cross-track residuals
    dt, dr = optimize.track_residuals(newtle, d)

    print(f"{newtle.line0}\n{newtle.line1}\n{newtle.line2}\n")
    print(prefit_rms, postfit_rms, optimize.rms(dt), optimize.rms(dr))
