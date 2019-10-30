#!/usr/bin/env python3
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, FK4, FK5, ICRS
from tlegenerator.iod import decode_iod_observation
import astropy.units as u
import ephem
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import optimize
import time

class two_line_element:
    """TLE class"""

    def __init__(self, tle0, tle1, tle2):
        """Define a tle"""

        self.tle0 = tle0 #.decode("utf-8")
        self.tle1 = tle1 #.decode("utf-8")
        self.tle2 = tle2 #.decode("utf-8")
        if self.tle0[:2]=="0 ":
            self.name = self.tle0[2:].strip()
        else:
            self.name = self.tle0.strip()
        self.id = self.tle1.split(" ")[1][:5]

    def __repr__(self):
        return "%s\n%s\n%s"%(self.tle0, self.tle1, self.tle2)
        
def set_checksum(line):
    s = 0
    for c in line[:-1]:
        if c.isdigit():
            s += int(c)
        if c == "-":
            s += 1
    return line[:-1]+"%d"%(s%10)        

def format_tle(satno, epoch, a, name="OBJ", desig="19600A  "):
    # Format B* drag term
    bstar_exp = int(np.log(np.abs(a[6]))/np.log(10.0))
    bstar_mantissa = (a[6]/10**bstar_exp)*1e5
    bstar = "%6d%2d"%(bstar_mantissa, bstar_exp)
    
    # Format TLE
    tle0 = name
    tle1 = set_checksum("1 %05dU %8s %14.8lf  .00000000  00000-0 %8s 0    08"%(satno, desig, epoch, bstar))
    tle2 = set_checksum("2 %05d %8.4f %8.4f %07.0f %8.4f %8.4f %11.8lf    08"%(satno, a[0], a[1], a[2]*1e6, a[3], a[4], a[5]))
    
    return two_line_element(tle0, tle1, tle2)
    
def residuals(a, epoch, observer, observations):
    # Format TLE
    tle = format_tle(99999, epoch, a)

    # Set satellite
    satellite = ephem.readtle(str(tle.tle0),
                              str(tle.tle1),
                              str(tle.tle2))
    
    # Loop over observations
    res = []
    for o in observations:
        observer.date = ephem.date(o.t.datetime)

        satellite.compute(observer)

        psat = SkyCoord(ra=float(satellite.ra), dec=float(satellite.dec), unit="rad", frame="fk5", equinox=o.t).transform_to(FK5(equinox="J2000"))
        res.append(psat.separation(o.p).degree)

    return np.array(res)

def chisq(a, epoch, observer, observations):
    res = residuals(a, epoch, observer, observations)
    return np.sum(res**2)

if __name__ == "__main__":
    # Open file
    fp = open("fit.dat", "r")
    lines = fp.readlines()
    fp.close()

    # Decode observations
    observations = [decode_iod_observation(line) for line in lines]
    
    # Set observer
    observer = ephem.Observer()
    observer.lon = "6.3785"
    observer.lat = "52.8344"
    observer.elevation = 10

    # Parameters
    a = [63.400, 40.9417, 0.10, 0.0, 130.0, 13.40779379, 0.5e-4]
    epoch = 19133.91247694

    print(format_tle(99999, epoch, a))
    print(np.std(residuals(a, epoch, observer, observations)))
    print(chisq(a, epoch, observer, observations))
    for i in range(5):
        start_time = time.time()
        q = optimize.fmin(chisq, a, args=(epoch, observer, observations))
        print(time.time()-start_time)
        print(format_tle(37386, epoch, q))
        print(np.std(residuals(q, epoch, observer, observations)))
        print(chisq(q, epoch, observer, observations))
        a = q