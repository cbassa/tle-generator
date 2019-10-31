#!/usr/bin/env python3

import argparse
import configparser
from pathlib import Path

from tlegenerator.iod import is_iod_observation, decode_iod_observation


def ingest_observations(observations_path, newlines):
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
            o = decode_iod_observation(newline)
            
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
    parser = argparse.ArgumentParser(description='Import observations file into the common file structure.')
    parser.add_argument('OBSERVATIONS_FILE', type=str,
                        help='File with observations in IOD format')
    args = parser.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read('configuration.ini')

    with open(args.OBSERVATIONS_FILE, errors="replace") as f:
        newlines = f.readlines()
        ingest_observations(cfg.get('Common', 'observations_path'), newlines)
