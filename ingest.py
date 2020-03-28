#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import configparser
from pathlib import Path

from tlegenerator import database as db
from tlegenerator import formats as fmt
from tlegenerator import observation as obs
from tlegenerator import twoline

def parse_observations(lines, observers, identifiers):
    """
    Reads a list of observation strings and writes them into the common file structure.
    """

    # Set up values for RDE format
    is_rde = False
    has_rde_date = False
    
    # Loop over lines
    observations = []
    for line in lines:
        # Clean line
        line = line.replace("\xa0", " ").rstrip()

        # Check if this line is an observation in the IOD format
        if fmt.is_iod_observation(line):
            # Decode IOD observation
            o = fmt.decode_iod_observation(line, observers)
        # Check if this line is an observation in the UK format
        elif fmt.is_uk_observation(line):
            # Decode UK observation
            o = fmt.decode_uk_observation(line, observers, identifiers)
        # Check if this line is a preamble to an observation in the RDE format
        elif fmt.is_rde_preamble(line):
            is_rde = True
            rde_preamble = line
            rde_date = None
            continue
        # Check if this line is the date of an observation in the RDE format
        elif fmt.is_rde_date(line) and is_rde:
            rde_date = int(line)
            has_rde_date = True
            continue
        # Check if this line is an observation in the RDE format
        elif fmt.is_rde_observation(line) and is_rde and has_rde_date:
            o = fmt.decode_rde_observation(rde_preamble, rde_date, line, observers, identifiers)
        # Check if this line signals the end of an RDE observation report
        elif fmt.is_rde_end(line) and is_rde:
            is_rde = False
            has_rde_date = False
            continue
        # Skip otherwise
        else:
            continue

        # Append to list
        if o is not None:
            observations.append((o.satno, o.desig_year, o.desig_id, o.site_id, o.t.datetime, o.iod_line, o.obs_condition, o.st, o.sp, o.p.ra.deg, o.p.dec.deg, o.epoch, o.uk_line, o.rde_preamble, o.rde_date, o.rde_line))

    return observations

def parse_elements(lines, origin=None):
    # Loop over line
    elements = []
    for i in range(1, len(lines)):
        if (lines[i][0]=="2") and (lines[i-1][0]=="1"):
            tle = twoline.TwoLineElement(lines[i-2], lines[i-1], lines[i])

            elements.append((tle.satno,
                             tle.desig_year,
                             tle.desig_id,
                             tle.name,
                             tle.line0,
                             tle.line1,
                             tle.line2,
                             tle.epoch,
                             tle.epochyr,
                             tle.epochdoy,
                             tle.classification,
                             tle.ndot,
                             tle.nddot,
                             tle.ephtype,
                             tle.elnum,
                             tle.incl,
                             tle.node,
                             tle.ecc,
                             tle.argp,
                             tle.m,
                             tle.n,
                             tle.revnum,
                             origin))
    return elements
 

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Import observations file into the common file structure.")
    parser.add_argument("-d", "--data", type=str,
                        help="Observations to ingest",
                        metavar="FILE")
    parser.add_argument("-c", "--catalog", type=str,
                        help="TLE catalog to ingest",
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

    # Input checking
    if not os.path.exists(args.conf_file):
        logger.error(f"{args.conf_file} not found")
        sys.exit()
    if args.data is None and args.catalog is None:
        parser.print_help()
        sys.exit()
    if args.data is not None and not os.path.exists(args.data):
        logger.error(f"{args.data} not found")
        sys.exit()
    if args.catalog is not None and not os.path.exists(args.catalog):
        logger.error(f"{args.catalog} not found")
        sys.exit()

    # Read configuration file
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    conf_file = args.conf_file
    cfg.read(conf_file)
    logger.info(f"Using config: {conf_file}")

    # Create database connection
    conn = db.create_connection(cfg.get("Common", "database_file"))

    # create tables
    if conn is not None:
        # Create tables
        db.create_table(conn, db.sql_create_observations_table)
        db.create_table(conn, db.sql_create_elements_table)
    else:
        logging.error("Cannot create the database connection.")
        sys.exit()

    # Parse observations
    if args.data is not None:
        # Read observers
        logger.info(f"Reading observers from {cfg.get('Common', 'observers_file')}")
        observers = obs.read_observers(cfg.get("Common", "observers_file"))

        # Read identifiers
        logger.info(f"Reading observers from {cfg.get('Common', 'identifiers_file')}")
        identifiers = fmt.read_identifiers(cfg.get("Common", "identifiers_file"))
    
        # Parsing observations
        logger.info(f"Parsing {args.data}")
        with open(args.data, errors="replace") as f:
            lines = f.readlines()
            observations = parse_observations(lines, observers, identifiers)

        # Insert into database
        with conn:
            cur = conn.cursor()
            cur.executemany(db.sql_insert_observations, observations)
            logger.info(f"Inserted {len(observations)} observations")

    # Parse TLEs
    if args.catalog is not None:
        # Parsing TLEs
        logger.info(f"Parsing {args.catalog}")
        with open(args.catalog, errors="replace") as f:
            lines = f.readlines()
            elements = parse_elements(lines)

        # Insert into database
        with conn:
            cur = conn.cursor()
            cur.executemany(db.sql_insert_elements, elements)
            logger.info(f"Inserted {len(elements)} TLEs")
