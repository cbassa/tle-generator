#!/usr/bin/env python3
import sys
import argparse
import configparser
import logging
from astropy.time import Time
from tlegenerator.observation import read_observers
from tlegenerator.update import update_tle

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
    parser.add_argument("-t", "--endtime",
                        help="Use observations upto this time (YYYY-MM-DDTHH:MM:SS) [default: now]")
    parser.add_argument("-m", "--endmjd",
                        help="Use observations upto this time (MJD; Modified Julian Day) [default: now]")
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
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    # Read observers
    logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
    observers = read_observers(cfg.get("Common", "observers_file"))

    # Find end time
    if (args.endtime is None) and (args.endmjd is None):
        tend = Time.now()
    elif (args.endtime is None) and (args.endmjd is not None):
        tend = Time(args.endmjd, format="mjd")
    else:
        tend = Time(args.endtime, format="isot", scale="utc")
 
    # Update TLE
    update_tle(args.ident, args.catalog, args.data, observers, tend, args.length)
