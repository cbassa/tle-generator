#!/usr/bin/env python3
import re
import logging
import warnings
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, FK4, FK5, ICRS
import astropy.units as u

class Observation:
    """Observation class"""

    def __init__(self, satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line, observer):
        self.satno = satno
        self.desig_year = desig_year
        self.desig_id = desig_id
        self.site_id = site_id
        self.obs_condition = obs_condition
        self.t = t
        self.st = st
        self.p = p
        self.sp = sp
        self.angle_format = angle_format
        self.epoch = epoch
        self.iod_line = iod_line.rstrip()
        self.observer = observer

    def __repr__(self):
        return self.iod_line
        
class Observer:
    """Observer class"""

    def __init__(self, site_id, lat, lon, elev, name):
        self.site_id = site_id
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.name = name

    def __repr__(self):
        return f"{self.site_id} {self.lat} {self.lon} {self.elev} {self.name}"

def decode_observer(line):
    site_id = int(line[0:4])
    lat = float(line[8:17])
    lon = float(line[18:27])
    elev = float(line[28:34])
    name = line[38:].rstrip()

    return Observer(site_id, lat, lon, elev, name)

        
def read_observers(fname):
    observers = []
    with open(fname, "r") as fp:
        lines = fp.readlines()

        for line in lines:
            if "#" in line:
                continue

            observers.append(decode_observer(line))

    return observers
            
        
def is_iod_observation(line):
    iod_pattern = r"\d{5} \d{2} \d{3}... \d{4} . \d{17} \d{2} \d{2} \d{7}.\d{6} \d{2}"

    if re.match(iod_pattern, line) is not None:
        return True
    else:
        return False
        
def insert_into_string(s, i, c):
    """Insert character c into string s at index i"""

    return s[:i] + c + s[i:]

def decode_HHMMSSs(s):
    s = insert_into_string(s, 8, "s")
    s = insert_into_string(s, 6, ".")
    s = insert_into_string(s, 4, "m")
    s = insert_into_string(s, 2, "h")
    return Angle(s)

def decode_HHMMmmm(s):
    s = insert_into_string(s, 8, "m")
    s = insert_into_string(s, 4, ".")
    s = insert_into_string(s, 2, "h")
    return Angle(s)

def decode_DDDMMSS(s):
    s = insert_into_string(s, 8, "s")
    s = insert_into_string(s, 5, "m")
    s = insert_into_string(s, 3, "d")
    return Angle(s)

def decode_DDDMMmm(s):
    s = insert_into_string(s, 8, "m")
    s = insert_into_string(s, 5, ".")
    s = insert_into_string(s, 3, "d")
    return Angle(s)

def decode_DDDdddd(s):
    s = insert_into_string(s, 8, "d")
    s = insert_into_string(s, 3, ".")
    return Angle(s)

def decode_DDMMSS(s):
    s = insert_into_string(s, 7, "s")
    s = insert_into_string(s, 5, "m")
    s = insert_into_string(s, 3, "d")
    return Angle(s)

def decode_DDMMmm(s):
    s = insert_into_string(s, 7, "m")
    s = insert_into_string(s, 5, ".")
    s = insert_into_string(s, 3, "d")
    return Angle(s)

def decode_DDdddd(s):
    s = insert_into_string(s, 7, "d")
    s = insert_into_string(s, 3, ".")
    return Angle(s)

def decode_iod_observation(iod_line, observers):
    # NORAD catalog ID
    satno = int(iod_line[0:5])

    # International designator
    desig_year, desig_id = iod_line[6:8], iod_line[9:15].rstrip()

    # Site identifier
    site_id = int(iod_line[16:20])

    # Observing conditions
    obs_condition = iod_line[21]

    # Decode time stamp
    timestamp = iod_line[23:40]
    timestamp = insert_into_string(timestamp, 14, ".")
    timestamp = insert_into_string(timestamp, 12, ":")
    timestamp = insert_into_string(timestamp, 10, ":")
    timestamp = insert_into_string(timestamp, 8, "T")
    timestamp = insert_into_string(timestamp, 6, "-")
    timestamp = insert_into_string(timestamp, 4, "-")

    # Set up to catch warnings
    with warnings.catch_warnings(record=True) as w:
        try:
            t = Time(timestamp, format="isot", scale="utc")
        except ValueError:
            logging.debug("Failed to decode timestamp")                              
            return None

        # Decode time uncertainty
        me, xe = int(iod_line[41]), int(iod_line[42])
        st = me*10.0**(xe - 8)

        # Decode angle format and epoch
        angle_format, epoch = int(iod_line[44]), int(iod_line[45])

        # Discard bad epochs
        if epoch!=5:
            logging.debug("Epoch not implemented")
            return None

        # Parse positional error
        me, xe = int(iod_line[62]), int(iod_line[63])
        sp = me * 10**(xe - 8)
        
        # Decode angles
        p = None
        angle1 = iod_line[47:54]
        angle2 = iod_line[54:61]
        if angle_format == 1:
            # Format 1: RA/DEC = HHMMSSs+DDMMSS MX   (MX in seconds of arc)
            try:
                ra = decode_HHMMSSs(angle1)
                dec = decode_DDMMSS(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
            sp = sp / 3600
        elif angle_format == 2:
            # Format 2: RA/DEC = HHMMmmm+DDMMmm MX   (MX in minutes of arc)
            try:
                ra = decode_HHMMmmm(angle1)
                dec = decode_DDMMmm(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
            sp = sp / 60
        elif angle_format == 3:
            # Format 3: RA/DEC = HHMMmmm+DDdddd MX   (MX in degrees of arc)
            try:
                ra = decode_HHMMmmm(angle1)
                dec = decode_DDdddd(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
        elif angle_format == 4:
            logging.debug("Format not implemented")
            # Format 4: AZ/EL  = DDDMMSS+DDMMSS MX   (MX in seconds of arc)
            az = decode_DDDMMSS(angle1)
            alt = decode_DDMMSS(angle2)
            p = None
            sp = sp / 3600
        elif angle_format == 5:
            logging.debug("Format not implemented")
            # Format 5: AZ/EL  = DDDMMmm+DDMMmm MX   (MX in minutes of arc)
            az = decode_DDMMmm(angle1)
            alt = decode_DDMMmm(angle2)
            p = None
            sp = sp / 60
        elif angle_format == 6:
            logging.debug("Format not implemented")
            # Format 6: AZ/EL  = DDDdddd+DDdddd MX   (MX in degrees of arc)
            az = decode_DDDdddd(angle1)
            alt = decode_DDdddd(angle2)
            p = None
        elif angle_format == 7:
            # Format 7: RA/DEC = HHMMSSs+DDdddd MX   (MX in degrees of arc)
            try:
                ra = decode_HHMMSSs(angle1)
                dec = decode_DDdddd(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
        else:
            logging.debug("Format not defined")
            return None

    # Discard observations with warnings
    if len(w)>0:
        logging.debug(str(w[-1].message))
        return None

    # Find observer
    found = False
    for observer in observers:
        if observer.site_id == site_id:
            found = True
            break

    # Discard observations with missing site information
    if not found:
        logging.debug(f"Site information missing for {site_id}")
        return None
    
    # Format observation
    o = Observation(satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line, observer)
        
    return o
