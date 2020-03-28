#!/usr/bin/env python3
import numpy as np
import astropy._erfa as erfa
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

class Dataset:
    def __init__(self, observations):
        # Get MJD timestamps
        mjd = np.array([o.t.mjd for o in observations])

        # Sort on time
        idx = np.argsort(mjd)
        
        self.tobs = Time(mjd[idx], format="mjd")
        self.site_id = np.array([o.site_id for o in observations])[idx]
        self.perr = np.array([o.sp for o in observations])[idx]
        self.terr = np.array([o.st for o in observations])[idx]
        lat = np.array([o.observer.lat for o in observations])[idx] * u.deg
        lon = np.array([o.observer.lon for o in observations])[idx] * u.deg
        elev = np.array([o.observer.elev for o in observations])[idx] * u.m
        r, v = EarthLocation(lat=lat, lon=lon, height=elev).get_gcrs_posvel(self.tobs)
        self.robs = (r.get_xyz().to(u.km).value).T
        self.vobs = (v.get_xyz().to(u.km / u.s).value).T
        self.R = teme_to_gcrs_matrix(self.tobs)
        self.raobs = np.array([o.p.ra.rad for o in observations])[idx]
        self.decobs = np.array([o.p.dec.rad for o in observations])[idx]
        self.uobs = np.array([np.cos(self.raobs) * np.cos(self.decobs),
                              np.sin(self.raobs) * np.cos(self.decobs),
                              np.sin(self.decobs)]).T
        self.mask = np.ones(len(mjd), dtype="bool")
        self.weight = np.ones(len(mjd), dtype="float")
        
class Observation:
    """Observation class"""

    def __init__(self, satno, desig_year, desig_id, site_id, obs_condition, t, st, p, sp, angle_format, epoch, iod_line, uk_line, rde_preamble, rde_date, rde_line, observer):
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
        self.uk_line = uk_line.rstrip()
        self.rde_preamble = rde_preamble.rstrip()
        self.rde_date = rde_date
        self.rde_line = rde_line.rstrip()
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

def rotation_matrix(theta, axis):
    # Angles
    ct, st = np.cos(theta), np.sin(theta)
    zeros, ones = np.zeros_like(theta), np.ones_like(theta)
    if axis == "x":
        R = np.array([[ones, zeros, zeros], [zeros, ct, -st], [zeros, st, ct]])
    elif axis == "y":
        R = np.array([[ct, zeros, st], [-st, zeros, ct], [zeros, ones, zeros]])
    elif axis == "z":
        R = np.array([[ct, -st, zeros], [st, ct, zeros], [zeros, zeros, ones]])

    if len(R.shape) == 3:
        return np.moveaxis(R, 2, 0)
    elif len(R.shape) == 2:
        return R

def teme_to_gcrs(t, pteme, vteme):
    # Equation of the equinoxes
    eqeq = erfa.ee00a(t.tt.jd, 0.0)
    Req = rotation_matrix(eqeq, "z")

    # Precession, nutation and bias
    Rpnb = erfa.pnm00a(t.tt.jd, 0.0)

    # Multiply matrices (take transpose of Rpnb)
    R = np.einsum("...ij,...kj->...ik", Req, Rpnb)

    # Multiply vectors
    pgcrs = np.einsum("i...jk,i...k->i...j", R, pteme)
    vgcrs = np.einsum("i...jk,i...k->i...j", R, vteme)
    
    return pgcrs, vgcrs

def teme_to_gcrs_matrix(t):
    # Equation of the equinoxes
    eqeq = erfa.ee00a(t.tt.jd, 0.0)
    Req = rotation_matrix(eqeq, "z")

    # Precession, nutation and bias
    Rpnb = erfa.pnm00a(t.tt.jd, 0.0)

    # Multiply matrices (take transpose of Rpnb)
    R = np.einsum("...ij,...kj->...ik", Req, Rpnb)

    return R
