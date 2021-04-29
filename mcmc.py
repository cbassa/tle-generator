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

from sgp4.api import Satrec, SatrecArray

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
    parser.add_argument("-n", "--steps",
                        help="Number of MCMC steps. [default: 10000]",
                        type=int, default=10000)
    parser.add_argument("-d", "--discard",
                        help="Number of MCMC steps to discard at start of chain. [default: 2000]",
                        type=int, default=2000)
    parser.add_argument("-t", "--thin",
                        help="MCMC thinning factor. [default: 10]",
                        type=int, default=10)
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
    if args.discard >= args.steps:
        logger.error("Discarding too many steps")
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
    sampler.run_mcmc(pos, args.steps, progress=True);

    # Plot walkers
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    
    labels = [r"$i$", r"$\Omega$", r"$e$", r"$\omega$", r"$M$", r"$n_0$", r"$B^{*}$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
    plt.savefig(f"{tle.satno:05d}_walkers.png")

    # Plot corner
    flat_samples = sampler.get_chain(discard=args.discard, thin=args.thin, flat=True)
    fig = corner.corner(flat_samples, labels=labels, truths=a, quantiles=(0.16, 0.5, 0.84))
    plt.savefig(f"{tle.satno:05d}_corner.png", dpi=70)

    # Compute parameters
    q = np.median(flat_samples, axis=0)
    sq = np.std(flat_samples, axis=0)
    postfit_rms = optimize.rms(optimize.residuals(q, tle.satno, tle.epochyr, tle.epochdoy, d))

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
            f.write(f"{newtle.line0}\n{newtle.line1}\n{newtle.line2}\n# {optimize.format_time_for_output(np.min(d.tobs[d.mask]))}-{optimize.format_time_for_output(np.max(d.tobs[d.mask]))}, {np.sum(d.mask)} obs, {optimize.rms(dt):.4f} sec, {optimize.rms(dr):.4f} deg rms\n")

    # Compute SGP4 position and velocity for reference TLE
    aref = np.array([tle.incl, tle.node, tle.ecc, tle.argp, tle.m, tle.n, tle.bstar])
    refsat = Satrec.twoline2rv(tle.line1, tle.line2)
    jd = refsat.jdsatepoch + refsat.jdsatepochF
    jdint = np.floor(jd)
    jdfrac = jd - jdint

    e, rsat, vsat = refsat.sgp4(jdint, jdfrac)
    qref = np.concatenate([rsat, vsat])

    # Compute SGP4 position and velocity of MCMC chain
    sats = []
    for f in flat_samples:
        line0, line1, line2 = twoline.format_tle(tle.satno, tle.epochyr, tle.epochdoy, *f, tle.name, tle.desig, classification="M")
        tle = twoline.TwoLineElement(line0, line1, line2)
        sats.append(Satrec.twoline2rv(tle.line1, tle.line2))
    sat = SatrecArray(sats)
    jdint = np.array([jdint])
    jdfrac = np.array([jdfrac])
    e, rsat, vsat = sat.sgp4(jdint, jdfrac)

    # Compute vector differences
    flat_samples = np.hstack([rsat.squeeze(), vsat.squeeze()]) - qref
    qref = np.zeros(6)

    # Convert velocities to m/s
    for i in range(3, 6):
        flat_samples[:, i] *= 1000

    # Report results
    q, sq = np.mean(flat_samples, axis=0), np.std(flat_samples, axis=0)
    text = f"{tle.satno:05d}"
    for i in range(6):
        text = text + f"{q[i]:7.3f}+-{sq[i]:.3f} "
    print(text + f"{np.sum(d.mask)} {optimize.rms(dt):.4f} {optimize.rms(dr):.4f}")
        
    # Create corner plot
    #labels = [r"$x$ (km)", r"$y$ (km)", r"$z$ (km)", r"$v_x$ (km s$^{-1}$)", r"$v_y$ (km s$^{-1}$)", r"$v_z$ (km s$^{-1}$)"]
    labels = [r"$\Delta x$ (km)", r"$\Delta y$ (km)", r"$\Delta z$ (km)", r"$\Delta v_x$ (m s$^{-1}$)", r"$\Delta v_y$ (m s$^{-1}$)", r"$\Delta v_z$ (m s$^{-1}$)"]
    fig = corner.corner(flat_samples, labels=labels, truths=qref, quantiles=(0.16, 0.5, 0.84))
    plt.savefig(f"{tle.satno:05d}_corner_cart_delta.png", dpi=70)
