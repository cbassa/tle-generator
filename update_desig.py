#!/usr/bin/env python3
import sys
import argparse
import configparser
import logging
import json
from spacetrack import SpaceTrackClient

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Update file with NORAD catalogue IDs and International " +
                                     "designations")
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

    # Download satellite catalog
    logger.info("Downloading satellite catalog")
    st = SpaceTrackClient(cfg.get("SpaceTrack", "username"),
                          cfg.get("SpaceTrack", "password"))
    satcat = json.loads(st.satcat(orderby="norad_cat_id", format="json"))

    # Store identifiers
    fname = cfg.get("Common", "identifiers_file")
    logger.info(f"Storing identifiers in {fname}")
    with open(fname, "w") as fp:
        for s in satcat:
            desig = s["OBJECT_ID"]
            satno = int(s["OBJECT_NUMBER"])
            fp.write(f"{satno:05d} {desig[2:4]}{desig[5:]}\n")
