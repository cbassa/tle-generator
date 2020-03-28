#!/usr/bin/env python3
import sqlite3
from sqlite3 import Error
import logging

sql_create_observations_table = """ CREATE TABLE IF NOT EXISTS observations (
satno integer NOT NULL,
desig_year text NOT NULL,
desig_id text NOT NULL,
site_id integer NOT NULL,
date text,
obs_condition text,
terr real,
perr real,
ra real,
dec real,
epoch int,
iod_line text PRIMARY KEY,
uk_line text,
rde_preamble text,
rde_date text,
rde_line text
); """

sql_create_elements_table = """ CREATE TABLE IF NOT EXISTS elements (
satno integer NOT NULL,
desig_year text NOT NULL,
desig_id text NOT NULL,
name text,
line0 text,
line1 text PRIMARY KEY,
line2 text,
epoch text,
epochyr integer,
epochdoy real,
classification text,
ndot real,
nddot real,
ephtype text,
elnum integer,
incl real,
node real,
ecc real,
argp real,
m real,
n real,
revnum integer,
origin text
); """

sql_insert_observations = """INSERT OR REPLACE INTO observations(satno, desig_year, desig_id, site_id, date, iod_line, obs_condition, terr, perr, ra, dec, epoch, uk_line, rde_preamble, rde_date, rde_line) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

sql_insert_elements = """INSERT OR REPLACE INTO elements(satno, desig_year, desig_id, name, line0, line1, line2, epoch, epochyr, epochdoy, classification, ndot, nddot, ephtype, elnum, incl, node, ecc, argp, m, n, revnum, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        logging.info(f"Opening connection to {db_file}")
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        logging.error(e)
 
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        logging.info(f"Creating table {create_table_sql.split(' ')[6]}")
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        logging.error(e)
