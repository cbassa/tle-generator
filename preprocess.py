#!/usr/bin/env python3
import sys
import argparse
import logging
import configparser
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
    parser = argparse.ArgumentParser(description='Preprocess observations.')
    parser.add_argument('-c', '--conf_file',
                        help="Specify configuration file. If no file" +
                        " is specified 'configuration.ini' is used.",
                        metavar="FILE")
    parser.add_argument('OBSERVATIONS_FILE', type=str,
                        help='File with observations in IOD format')
    args = parser.parse_args()

    # Read configuration file
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
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
    observers = read_observers(cfg.get('Common', 'observers_file'))

    # Read observations
    f = open(args.OBSERVATIONS_FILE, errors="replace")
    newlines = f.readlines()
    f.close()

    # Parse observations
    observations = [decode_iod_observation(newline, observers) for newline in newlines]

    # Restructure
    d = Dataset(observations)

    tnew = Time("2020-01-08T00:00:00", format="isot", scale="utc")
    tlength = 40 * u.d
    d.mask = (d.tobs >= tnew - tlength) & (d.tobs < tnew)
    imax = np.argmax(d.tobs[d.mask].mjd)
    tnew = d.tobs[imax]
    
    # Read tle
    fp = open("37386.txt", "r")
    lines = fp.readlines()
    fp.close()
    tle = TwoLineElement(lines[0], lines[1], lines[2])

    # Propagate
    newepoch = tnew.datetime
    newtle, converged = propagate(tle, newepoch)
    
    # Extract parameters
    a = np.array([newtle.incl, newtle.node, newtle.ecc, newtle.argp, newtle.m, newtle.n, newtle.bstar])

    # Optimize
    for i in range(10):
        p = optimize.fmin(chisq, a, args=(newtle.satno, newtle.epochyr, newtle.epochdoy, d), disp=False)
        r = residuals(p, newtle.satno, newtle.epochyr, newtle.epochdoy, d)
        print(i, rms(r))
        a = p
    print(tnew.isot)
    print(rms(r))
    print(len(r), np.sum(d.mask))

    line0, line1, line2 = format_tle(newtle.satno, newtle.epochyr, newtle.epochdoy, *p, newtle.name, newtle.desig)
    print(f"{line0}\n{line1}\n{line2}")

    for i in range(len(observations)):
        if d.mask[i]:
            print(observations[i].iod_line)
