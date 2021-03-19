#!/usr/bin/env python3
import os
import sys
import argparse
import configparser
import logging
import yaml

import astropy.units as u
from astropy.time import Time

from tlegenerator import observation as obs
from tlegenerator import formats as fmt
from tlegenerator import database as db
from tlegenerator import twoline
from tlegenerator import update

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Preprocess observations.")
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
    if args.ident is None:
        logger.error("Provide NORAD ID to update")
        sys.exit()
    else:
        satno = int(args.ident)
        
    # Read configuration file
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(args.conf_file)

    # Read observers
    logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
    observers = obs.read_observers(cfg.get("Common", "observers_file"))

    # Parse end time
    if (args.endtime is None) and (args.endmjd is None):
        tend = Time.now()
    elif (args.endtime is None) and (args.endmjd is not None):
        tend = Time(args.endmjd, format="mjd")
    else:
        tend = Time(args.endtime, format="isot", scale="utc")
    
    # Open database
    conn = db.create_connection(cfg.get("Common", "database_file"))

    # Select last observation before end time
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM observations WHERE date <= ? AND satno = ?", (tend.datetime, satno))
    row = cur.fetchall()[0]

    # Set time range
    if row[0] is None:
        logger.error(f"Error selecting observations for {satno} before {tend.isot}")
        sys.exit()
    else:
        tmax = Time(row[0], format="iso", scale="utc")
        tmin = tmax - args.length * u.d

    # Select observations in time range
    cur.execute("SELECT iod_line FROM observations WHERE date > ? AND date <= ? AND satno = ? ORDER BY date", (tmin.datetime, tmax.datetime, satno))
    rows = cur.fetchall()
    logger.info(f"Selected {len(rows)} observations of {satno} between {tmin.isot} and {tmax.isot}")

    data = {}
    
    # Process observations
    if len(rows) > 0:
        # Store observations
        observations = [fmt.decode_iod_observation(row[0], observers) for row in rows]

        # Store data
        with open(f"{satno:05d}.dat", "w") as f:
            for o in observations:
                f.write(f"{o.iod_line}\n")
    
        # Store yaml
        data = data | {"observations": [o.iod_line for o in observations]}
    
    # Select TLE
    cur = conn.cursor()
    cur.execute("SELECT line0,line1,line2 FROM elements WHERE epoch < ? AND satno = ? ORDER BY epoch DESC LIMIT 1", (tmax.datetime, satno))
    rows = cur.fetchall()

    # Process TLE
    if len(rows) > 0:
        row = rows[0]
        tle = twoline.TwoLineElement(row[0], row[1], row[2])
        
        data = data | {"tle": {"line0": tle.line0,
                               "line1": tle.line1,
                               "line2": tle.line2}}
        
        # Store TLE
        with open(f"{satno:05d}.txt", "w") as f:
            f.write(f"{tle.line0}\n{tle.line1}\n{tle.line2}\n")

    # Store YAML
    with open(f"{satno:05d}.yaml", "w") as fp:
        yaml.dump(data, fp, sort_keys=True)
        
