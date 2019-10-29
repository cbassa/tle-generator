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

def plot():
    t = Time([o.t.mjd for o in observations], format="mjd")
    y = np.zeros_like(t)
    s = np.array([o.site_id for o in observations])

    tplot = mdates.date2num(t.datetime)
    
    fig, ax = plt.subplots(figsize=(15, 5))

    usite_id = np.unique(s)
    
    date_format = mdates.DateFormatter("%F")
    fig.autofmt_xdate(rotation=0, ha="center")

    for us in usite_id:
        c = s==us
        ax.scatter(tplot[c], y[c], alpha=0.3, label="%04d"%us)
    ax.set_ylabel("Brightness (ADU)")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(ncol=4)

    
    plt.tight_layout()
    plt.savefig("plot.png")

def residuals():
    # Set observer
    observer = ephem.Observer()
    observer.lon = "6.3785"
    observer.lat = "52.8344"
    observer.elevation = 10

    tle_lines = ["OBJ", "1 31701U 07027A   19132.86965740  .00000000  00000-0  50000-4 0    01", "2 31701  63.5301 110.4000 0001000   0.0000  91.0111 13.55100810    06"]

    tle = two_line_element(tle_lines[0], tle_lines[1], tle_lines[2])

    observer.date = ephem.date(t.datetime)

    satellite = ephem.readtle(str(tle.tle0),
                               str(tle.tle1),
                               str(tle.tle2))

    satellite.compute(observer)

    psat = SkyCoord(ra=float(satellite.ra), dec=float(satellite.dec), unit="rad", frame="fk5", equinox=t).transform_to(FK5(equinox="J2000"))

    print(psat)
    print(psat.separation(pobs))

    
        
if __name__ == "__main__":
    # Open file
    fp = open("37386.dat", "r")
    lines = fp.readlines()
    fp.close()

    # Decode observations
    observations = [decode_iod_observation(line) for line in lines]

    t = Time([o.t.mjd for o in observations], format="mjd")
    tmin, tmax = np.min(t), np.max(t)

    dt = 15.0
    
    for i, mjd0 in enumerate(np.arange(np.floor(tmin.mjd)+dt, np.floor(tmax.mjd))):
        c = (t.mjd>mjd0-dt) & (t.mjd<=mjd0)
        print(mjd0, np.sum(c))

        fp = open("p%04d.dat"%i, "w")
        for j, o in enumerate(observations):
            if c[j]:
                fp.write(o.iod_line)

        fp.close()
        
