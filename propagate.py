#!/usr/bin/env python3
"""
Test program for propagate a TLE to a new epoch (see issue #7).

Reference TLE
NOSS 3-6 (A)
1 38758U 12048A   19304.77896478 0.00000000  00000-0  00000-0 0    02
2 38758  63.4414 256.0512 0104915   1.4757 358.5242 13.40790232    00

sattools propagate output:
propagate -i 38758 -t 2019-11-06T12:00:00
1 38758U 12048A   19310.50000000  .00000000  00000-0  00000-0 0    08
2 38758  63.4414 241.4959 0104925   1.4436 253.0729 13.40790383    04
"""
import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from sgp4.ext import rv2coe, jday, invjday

def set_checksum(line):
    s = 0
    for c in line[:-1]:
        if c.isdigit():
            s += int(c)
        if c == "-":
            s += 1
    return line[:-1]+"%d"%(s%10)        

def date2doy(year, mon, day, hr, minute, sec):
    # Leap years
    if (year%4==0) and (year%400!=0):
        k = 1
    else:
        k = 2

    # Day of year
    doy = np.floor(275.0*mon/9.0)-k*np.floor((mon+9.0)/12.0)+day-30

    # Fractional day
    fday = hr/24.0+minute/3600.0+sec/86400.0
    
    return doy+fday

def format_tle_from_coe(coe, ep_year, ep_doy, sat, desig, bstarstr):
    # Radians to degrees
    r2d = 180.0/np.pi
    xpdotp = 1440.0/(2.0*np.pi)

    # Extract parameters
    p, a, ecc, incl, node, argp, nu, m, arglat, truelon, lonper = coe
    
    # Compute mean motion
    no = sat.whichconst.xke*(a/sat.whichconst.radiusearthkm)**(-3.0/2.0)*xpdotp

    # Format tle
    tle1 = "1 %05dU %-8s %02d%12.8lf 0.00000000  00000-0 %8s 0    08" % (sat.satnum, desig,
                                                                           ep_year-2000, ep_doy,
                                                                           bstarstr)
    tle2 = "2 %05d %8.4f %8.4f %07.0f %8.4f %8.4f %11.8lf    08" % (sat.satnum, incl*r2d,
                                                                    node*r2d, ecc*1e7,
                                                                    argp*r2d, m*r2d, no)

    return set_checksum(tle1), set_checksum(tle2)

def propagate_tle(date, intle1, intle2):
    # Read TLE
    sat = twoline2rv(intle1, intle2, wgs84)

    # Get reference state vector
    r0, v0 = sat.propagate(*date)

    # Exit if state vector contains nans
    if np.isnan(np.array(r0+v0)).any():
        return "", "", False
    
    # New epoch
    ep_year = date[0]
    ep_doy = date2doy(*date)

    # Get info
    desig = get_desig(intle1)
    bstarstr = get_bstarstr(intle1)

    # Initial COE (classical orbital elements)
    coe_init = np.array(rv2coe(r0, v0, sat.whichconst.mu))
    coe_prev = coe_init

    # Format into TLE
    tle1, tle2 = format_tle_from_coe(coe_init, ep_year, ep_doy, sat, desig, bstarstr)

    # Loop to iterate
    converged = False
    for k in range(100):
        # Propagate
        sat = twoline2rv(tle1, tle2, wgs84)
        r, v = sat.propagate(*date)

        # Vector differences
        dr = np.linalg.norm(np.array(r)-np.array(r0))
        dv = np.linalg.norm(np.array(v)-np.array(v0))
        
        # Updated COE
        coe_new = np.array(rv2coe(r, v, sat.whichconst.mu))

        # Adjust COE
        coe = coe_new
        for i in [1, 2, 3, 4, 5, 7]:
            coe[i] = coe_prev[i]+coe_init[i]-coe_new[i]

        # Keep inclination positive
        if coe[3]<0.0:
            coe[3]*=-1
            coe[4]+=np.pi
        if coe[3]>np.pi:
            coe[3] = np.pi
            
        # Keep eccentricity positive
        if coe[2]<0.0:
            coe[2] = 0.0

        # Keep angles between 0 and 2pi
        for i in [4, 5, 7]:
            coe[i] = np.mod(coe[i], 2.0*np.pi)
            
        # Format into TLE
        tle1, tle2 = format_tle_from_coe(coe, ep_year, ep_doy, sat, desig, bstarstr)

        # Save
        coe_prev = coe

        # Exit on convergence (dr in km, dv in km/s)
        if (dr<0.01) & (dv<0.001):
            converged = True
            break

    return tle1, tle2, converged

def get_desig(tle1):
    return tle1[9:17]

def get_bstarstr(tle1):
    return tle1[53:61]

if __name__ == "__main__":
    with open("catalog.tle", "r") as fp:
        lines = fp.readlines()

    # Propagate to
    date = [2019, 11, 6, 12, 0, 0]

    # Propagate
    for i in range(0, len(lines), 3):
        tle1, tle2, converged = propagate_tle(date, lines[i+1], lines[i+2])

        print(lines[i].rstrip())
        print(tle1)
        print(tle2)
            
