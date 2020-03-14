#!/usr/bin/env python3
import sys
import argparse
import logging
import configparser
from pathlib import Path

from tlegenerator.iod import is_iod_observation, decode_iod_observation, read_observers


def ingest_observations(observations_path, newlines, observers):
    '''
    Reads a list of IOD observation strings and writes them into the common file structure.
    '''

    # Check if IOD
    for newline in newlines:
        # Check if this is an IOD observation
        if is_iod_observation(newline):
            # Clean line
            newline = newline.replace("\xa0", " ")
            
            # Decode IOD observation
            o = decode_iod_observation(newline, observers)

            print(o.t.isot, o.observer.lat, o.observer.lon)
            
            # Skip bad observations
            if o is None:
                logger.debug("Discarding %s" % newline.rstrip())
                continue
            
            # Data file name
            fname = Path(observations_path, f"{o.satno:05d}.dat")

            # Read existing observations
            oldlines = []
            if fname.exists():
                with open(fname, "r") as f:
                    oldlines = f.readlines()
                    # NOTE: This might take a considerable amount of RAM
                    # if there are many obs as it reads all previous observations.

            # Append if no duplicate
            if not newline in oldlines:
                oldlines.append(newline)
                
            # Lines to write
            with open(fname, "w") as f:
                for line in oldlines:
                    f.write("%s" % line)


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

    logger.info("Using config: %s" % conf_file)

    # Read observers
    logger.info("Reading observers from %s" % cfg.get('Common', 'observers_file'))
    observers = read_observers(cfg.get('Common', 'observers_file'))

    # Parse observations
    logger.info("Parsing %s" % args.OBSERVATIONS_FILE)
    with open(args.OBSERVATIONS_FILE, errors="replace") as f:
        newlines = f.readlines()
        ingest_observations(cfg.get('Common', 'observations_path'), newlines, observers)
