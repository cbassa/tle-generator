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
import corner
import emcee

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
    tle = twoline.TwoLineElement(*list(data["tle"].values()))

    # Extract observations
    observations = [fmt.decode_iod_observation(line, observers) for line in data["observations"]]

    # Generate dataset
    d = obs.Dataset(observations)

    # New epoch time
    tmax = np.max(d.tobs)
    tmin = np.min(d.tobs)

    # Extract parameters
    a = np.array([tle.incl, tle.node, tle.ecc, tle.argp, tle.m, tle.n, tle.bstar])
    sa = np.array([0.0001, 0.0001, 0.000001, 0.0001, 0.0001, 0.00001, 1e-5])

    # Compute prefit RMS
    prefit_rms = optimize.rms(optimize.residuals(a, tle.satno, tle.epochyr, tle.epochdoy, d))
    
    # Intialize walkders
    pos = a + sa * np.random.randn(32, len(a))
    nwalkers, ndim = pos.shape

    # Run sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, optimize.log_probability, args=(tle.satno, tle.epochyr, tle.epochdoy, d))
    sampler.run_mcmc(pos, 10000, progress=True);

    # Plot walkers
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    # Keep in range
    for i in [1, 3, 4]:
        samples[:, :, i] = np.mod(samples[:, :, i], 360.0)
    
    labels = [r"$i$", r"$\Omega$", r"$e$", r"$\omega$", r"$M$", r"$n_0$", r"$B^{*}$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
    plt.savefig("walkers.png")

    #tau = sampler.get_autocorr_time()
    #print(tau)

    flat_samples = sampler.get_chain(discard=2000, thin=10, flat=True)

    q = np.median(flat_samples, axis=0)
    sq = np.std(flat_samples, axis=0)
#    for i in range(len(q)):
#        print("%d %f +- %f" % (i, q[i], sq[i]))
        
    postfit_rms = optimize.rms(optimize.residuals(q, tle.satno, tle.epochyr, tle.epochdoy, d))
        
    fig = corner.corner(flat_samples, labels=labels, truths=a)
    #fig.set_size_inches(12, 12)
    plt.savefig("corner.png", dpi=70)

    # Format TLE
    line0, line1, line2 = twoline.format_tle(tle.satno, tle.epochyr, tle.epochdoy, *q, tle.name, tle.desig, classification="M")
    newtle = twoline.TwoLineElement(line0, line1, line2)
    
    # Get in-track, cross-track residuals
    dt, dr = optimize.track_residuals(newtle, d)

    print(f"{newtle.line0}\n{newtle.line1}\n{newtle.line2}\n# {optimize.format_time_for_output(np.min(d.tobs[d.mask]))}-{optimize.format_time_for_output(np.max(d.tobs[d.mask]))}, {np.sum(d.mask)} obs, {optimize.rms(dt):.4f} sec, {optimize.rms(dr):.4f} deg rms")
    
    # Store yaml
    data = {"tle": {"line0": newtle.line0,
                    "line1": newtle.line1,
                    "line2": newtle.line2},
            "observations": [o.iod_line for o in observations]}
    
    with open(f"{tle.satno:05d}.yaml", "w") as fp:
        yaml.dump(data, fp, sort_keys=True)

    # Store TLE
    with open(f"{tle.satno:05d}.txt", "w") as f:
        f.write(f"{newtle.line0}\n{newtle.line1}\n{newtle.line2}\n")

    # Store samples
    with open(f"{tle.satno:05d}_mcmc.txt", "w") as f:
        for i in np.random.randint(0, flat_samples.shape[0], 100):
            line0, line1, line2 = twoline.format_tle(tle.satno, tle.epochyr, tle.epochdoy, *flat_samples[i], tle.name, tle.desig)
            newtle = twoline.TwoLineElement(line0, line1, line2)
            f.write(f"{tle.line0}\n{tle.line1}\n{tle.line2}\n# {optimize.format_time_for_output(np.min(d.tobs[d.mask]))}-{optimize.format_time_for_output(np.max(d.tobs[d.mask]))}, {np.sum(d.mask)} obs, {optimize.rms(dt):.4f} sec, {optimize.rms(dr):.4f} deg rms\n")
