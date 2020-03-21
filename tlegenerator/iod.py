#!/usr/bin/env python3
import re
import logging
import warnings
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, FK4, FK5, ICRS
import astropy.units as u
from tlegenerator.observation import Observation
           
def is_iod_observation(line):
    iod_pattern = r"\d{5} \d{2} \d{3}... \d{4} . \d{17} \d{2} \d{2} \d{7}.\d{6} \d{2}"

    if re.match(iod_pattern, line) is not None:
        return True
    else:
        return False

def is_uk_observation(line):
    uk_pattern = r"\d{20}"

    if re.match(uk_pattern, line) is not None:
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

def read_identifiers(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    d = []
    for line in lines:
        d.append({"satno": int(line.split(" ")[0]),
                  "desig": line.rstrip().split(" ")[1]})
    return d

def number_to_letter(n):
    x = (n - 1) % 24
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    rest = (n - 1) // 24
    if rest == 0:
        return letters[x]
    return number_to_letter(rest) + letters[x]

def decode_uk_observation(uk_line, observers, identifiers):
    # International designator
    desig_year, desig_num, desig_part = int(uk_line[0:2]), int(uk_line[2:5]), number_to_letter(int(uk_line[5:7]))
    desig_id = f"{desig_num:03d}{desig_part:s}"
    desig = f"{desig_year:02d}{desig_id}"
    
    # Find satno
    satno = 99999
    for ident in identifiers:
        if ident["desig"] == desig:
            satno = ident["satno"]

    # Site identifier
    site_id = int(uk_line[7:11])

    # Decode timestamp
    year = int(uk_line[11:13])
    month = int(uk_line[13:15])
    day = int(uk_line[15:17])
    hour = int(uk_line[17:19])
    minute = int(uk_line[19:21])
    sec = float(insert_into_string(uk_line[21:27], 2, "."))
    if year < 57:
        year += 2000
    else:
        year += 1900
    timestamp = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:06.3f}"
    iod_timestamp = f"{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{1000*sec:05.0f}"
    
    # Set up to catch warnings
    with warnings.catch_warnings(record=True) as w:
        try:
            t = Time(timestamp, format="isot", scale="utc")
        except ValueError:
            logging.debug("Failed to decode timestamp")                              
            return None

        # Time uncertainty
        st = float(insert_into_string(uk_line[27:31], 1, "."))

        # Time and angle format
        time_format, angle_format = int(uk_line[32]), int(uk_line[33])

        # Position uncertainty
        sp = float(insert_into_string(uk_line[50:53], 1, "."))

        # Epoch code
        epoch = int(uk_line[54])
        
        # Discard bad epochs
        if epoch!=5:
            logging.debug("Epoch not implemented")
            return None

        # Decode angles
        p = None
        angle1 = uk_line[34:42]
        angle2 = uk_line[42:49]
        if angle_format == 1:
            # Format 1: RA/DEC = HHMMSSss+DDMMSSs (seconds of arc, SSSs)
            try:
                ra = decode_HHMMSSs(angle1)
                dec = decode_DDMMSS(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
            sp = sp / 3600
        elif angle_format == 2:
           # Format 2: RA/DEC = HHMMmmmm+DDMMmmm (minutes of arc, MMmm)
            try:
                ra = decode_HHMMmmm(angle1)
                dec = decode_DDMMmm(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
            sp = sp / 60
        elif angle_format == 3:
           # Format 3: RA/DEC = HHMMmmmm+DDddddd (degrees of arc, Dddd)
            try:
                ra = decode_HHMMmmm(angle1)
                dec = decode_DDdddd(angle2)
                p = SkyCoord(ra=ra, dec=dec, frame=FK5)
            except:
                logging.debug("Failed to decode position")
                p = None
        else:
            logging.debug("Format not implemented")
            p = None

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

    # Encode time uncertainty
    tx = int(np.floor(np.log10(st)) + 8)
    tm = int(np.floor(st * 10**(-(tx - 8))))

    # Encode position uncertainty
    px = int(np.floor(np.log10(sp)) + 8)
    pm = int(np.floor(sp * 10**(-(px - 8))))

    # Set observing condition
    obs_condition = "G"
    
    # Encode position (assumes to_string returns HH:MM:SS.S +DD:MM:SS)
    rah, ram, ras = p.ra.hms
    ded, dem, des = p.dec.dms
    pstr = f"{rah:02.0f}{ram:02.0f}{10 * ras:03.0f}{ded:+03.0f}{np.abs(dem):02.0f}{np.abs(des):02.0f}"
    
    #pstr = p.to_string("hmsdms").replace("h", "").replace("m", "").replace("s", "").replace("d", "").replace(" ", "").replace(".", "")

    # Format IOD line
    iod_line = f"{satno:05d} {desig_year:02d} {desig_id:<6s} {site_id:04d} {obs_condition:1s} {iod_timestamp} {tm:1d}{tx:1d} 15 {pstr} {pm:1d}{px:1d}"
    # Format observation
    o = Observation(satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line, observer)
        
    return o

 
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
        st = me * 10.0**(xe - 8)

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
