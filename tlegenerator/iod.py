#!/usr/bin/env python3
import re
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, FK4, FK5, ICRS
import astropy.units as u

class Observation:
    """Observation class"""

    def __init__(self, satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line):
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
        self.iod_line = iod_line

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

def decode_iod_observation(iod_line):
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
    try:
        t = Time(timestamp, format="isot", scale="utc")
    except ValueError:
        t = None

    # Decode time uncertainty
    me, xe = int(iod_line[41]), int(iod_line[42])
    st = me*10.0**(xe-8)

    # Decode angle format and epoch
    angle_format, epoch = int(iod_line[44]), int(iod_line[45])

    if epoch!=5:
        print("Epoch not implemented!")
    
    # Decode angle
    angle1 = iod_line[47:54]
    angle2 = iod_line[54:61]
    if angle_format==1:
        # Format 1: RA/DEC = HHMMSSs+DDMMSS MX   (MX in seconds of arc)
        ra = decode_HHMMSSs(angle1)
        dec = decode_DDMMSS(angle2)
        p = SkyCoord(ra=ra, dec=dec, frame=FK5)
    elif angle_format==2:
        # Format 2: RA/DEC = HHMMmmm+DDMMmm MX   (MX in minutes of arc)
        ra = decode_HHMMmmm(angle1)
        dec = decode_DDMMmm(angle2)
        p = SkyCoord(ra=ra, dec=dec, frame=FK5)
    elif angle_format==3:
        # Format 3: RA/DEC = HHMMmmm+DDdddd MX   (MX in degrees of arc)
        ra = decode_HHMMmmm(angle1)
        dec = decode_DDdddd(angle2)
        p = SkyCoord(ra=ra, dec=dec, frame=FK5)
    elif angle_format==4:
        # Format 4: AZ/EL  = DDDMMSS+DDMMSS MX   (MX in seconds of arc)
        az = decode_DDDMMSS(angle1)
        alt = decode_DDMMSS(angle2)
    elif angle_format==5:
        # Format 5: AZ/EL  = DDDMMmm+DDMMmm MX   (MX in minutes of arc)
        az = decode_DDMMmmm(angle1)
        alt = decode_DDMMmm(angle2)
    elif angle_format==6:
        # Format 6: AZ/EL  = DDDdddd+DDdddd MX   (MX in degrees of arc)
        az = decode_DDDdddd(angle1)
        alt = decode_DDdddd(angle2)
    elif angle_format==7:
        # Format 7: RA/DEC = HHMMSSs+DDdddd MX   (MX in degrees of arc)
        ra = decode_HHMMSSs(angle1)
        dec = decode_DDdddd(angle2)
        p = SkyCoord(ra=ra, dec=dec, frame=FK5)
    else:
        print("Format not defined")

    sp = 0.0
        
    o = Observation(satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line)
        
    return o
