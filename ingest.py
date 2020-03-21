#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import configparser
from pathlib import Path

from tlegenerator.iod import is_iod_observation, decode_iod_observation
from tlegenerator.iod import is_uk_observation, decode_uk_observation
from tlegenerator.iod import read_identifiers
from tlegenerator.observation import read_observers

def ingest_observations(observations_path, newlines, observers, identifiers):
    '''
    Reads a list of observation strings and writes them into the common file structure.
    '''
    
    # Loop over lines
    for newline in newlines:
        # Clean line
        newline = newline.replace("\xa0", " ")

        # Check if this is an observation in IOD format
        if is_iod_observation(newline):
            # Decode IOD observation
            o = decode_iod_observation(newline, observers)
        # Check if this is an observation in UK format
        elif is_uk_observation(newline):
            # Decode UK observation
            o = decode_uk_observation(newline, observers, identifiers)
            if o is not None:
                o = decode_iod_observation(o.iod_line, observers)
        # Skip otherwise
        else:
            continue
            
        # Skip bad observations
        if o is None:
            logger.debug(f"Discarding {newline.rstrip()}")

            fname = Path(observations_path, "rejected.dat")
            iod_line = newline
        else:
            fname = Path(observations_path, f"{o.satno:05d}.dat")
            iod_line = o.iod_line

        # Read existing observations
        oldlines = []
        if fname.exists():
            with open(fname, "r") as f:
                oldlines = f.readlines()
                # NOTE: This might take a considerable amount of RAM
                # if there are many obs as it reads all previous observations.

        # Append if no duplicate
        if not iod_line in oldlines:
            oldlines.append(iod_line)
                
        # Lines to write
        with open(fname, "w") as f:
            for line in oldlines:
                f.write(f"{line.rstrip()}\n")


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description='Import observations file into the common file structure.')
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

    logger.info(f"Using config: {conf_file}")

    # Read observers
    logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
    observers = read_observers(cfg.get('Common', 'observers_file'))

    # Read identifiers
    logger.info(f"Reading observers from {cfg.get('Common', 'identifiers_file')}")
    identifiers = read_identifiers(cfg.get('Common', 'identifiers_file'))
    
    # Generate directory
    if not os.path.exists(cfg.get('Common', 'observations_path')):
        os.makedirs(cfg.get('Common', 'observations_path'))
    
    # Parse observations
    logger.info(f"Parsing {args.OBSERVATIONS_FILE}")
    with open(args.OBSERVATIONS_FILE, errors="replace") as f:
        newlines = f.readlines()
        ingest_observations(cfg.get('Common', 'observations_path'), newlines, observers, identifiers)
