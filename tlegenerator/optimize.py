#!/usr/bin/env python3
import numpy as np
from sgp4.api import Satrec
import astropy.units as u
from astropy.coordinates import SkyCoord
from tlegenerator.twoline import format_tle

def residuals(a, satno, epochyr, epochdoy, d):
    # Format TLE from parameters
    line0, line1, line2 = format_tle(satno, epochyr, epochdoy, *a)

    # Set up satellite
    sat = Satrec.twoline2rv(line1, line2)
    
    # Compute integer and fractional JD
    jdint = np.floor(d.tobs.jd[d.mask])
    jdfrac = d.tobs.jd[d.mask] - jdint
    
    # Evaluate SGP4
    e, rsat_teme, vsat_teme = sat.sgp4_array(jdint, jdfrac)

    # Convert to GCRS
    rsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], rsat_teme)
    vsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], vsat_teme)

    # Compute unit vectors
    dr = rsat - d.robs[d.mask]
    r = np.linalg.norm(dr, axis=1)
    upred = dr / r[:, np.newaxis]

    # Compute residuals (in degrees)
    res = np.arccos(np.sum(d.uobs[d.mask] * upred, axis=1)) * 180 / np.pi
    
    return res

def chisq(a, satno, epochyr, epochdoy, d):
    return np.sum((residuals(a, satno, epochyr, epochdoy, d) / d.perr[d.mask] * d.weight[d.mask])**2)


def log_likelihood(a, satno, epochyr, epochdoy, d):
    r = residuals(a, satno, epochyr, epochdoy, d)
    return -0.5 * np.sum(((r / d.perr[d.mask] * d.weight[d.mask]) ** 2 + np.log(2.0 * np.pi * d.perr[d.mask] ** 2)))


def log_likelihood_fixed_bstar(a, satno, epochyr, epochdoy, bstar, d):
    b = np.append(a, bstar)
    r = residuals(b, satno, epochyr, epochdoy, d)
    return -0.5 * np.sum(((r / d.perr[d.mask] * d.weight[d.mask]) ** 2 + np.log(2.0 * np.pi * d.perr[d.mask] ** 2)))


def log_prior(a):
    if 0.0 <= np.mod(a[0], 180.0) <= 180.0 and 0.0 <= np.mod(a[1], 360.0) < 360.0 and 0.0 < a[2] < 1.0 and 0.0 <= np.mod(a[3], 360.0) < 360.0 and 0.0 <= np.mod(a[4], 360.0) < 360.0 and 0.035 <= a[5] <= 18.0 and np.abs(a[6]) < 1e-2:
        return 0.0
    return -np.inf


def log_prior_fixed_bstar(a):
    if 0.0 <= np.mod(a[0], 180.0) <= 180.0 and 0.0 <= np.mod(a[1], 360.0) < 360.0 and 0.0 < a[2] < 1.0 and 0.0 <= np.mod(a[3], 360.0) < 360.0 and 0.0 <= np.mod(a[4], 360.0) < 360.0 and 0.035 <= a[5] <= 18.0:
        return 0.0
    return -np.inf


def log_probability(a, satno, epochyr, epochdoy, d):
    lp = log_prior(a)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(a, satno, epochyr, epochdoy, d)
    if np.isnan(ll):
        return -np.inf
    return lp + ll


def log_probability_fixed_bstar(a, satno, epochyr, epochdoy, bstar, d):
    lp = log_prior_fixed_bstar(a)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_fixed_bstar(a, satno, epochyr, epochdoy, bstar, d)
    if np.isnan(ll):
        return -np.inf
    return lp + ll


def rms(x):
    return np.sqrt(np.sum(x**2) / len(x))


def track_residuals(tle, d):
    # Observed positions
    pobs = SkyCoord(ra=d.raobs[d.mask], dec=d.decobs[d.mask], frame="icrs", unit="rad")
    
    # Set up satellite
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    
    # Compute integer and fractional JD
    jdint = np.floor(d.tobs.jd[d.mask])
    jdfrac = d.tobs.jd[d.mask] - jdint

    # Evaluate SGP4
    e, rsat_teme, vsat_teme = sat.sgp4_array(jdint, jdfrac)

    # Convert to GCRS
    rsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], rsat_teme)
    vsat = np.einsum("i...jk,i...k->i...j", d.R[d.mask], vsat_teme)

    # Compute sky coords
    dr = rsat - d.robs[d.mask]
    r = np.linalg.norm(dr, axis=1)
    ra = np.arctan2(dr[:, 1], dr[:, 0])
    dec = np.arcsin(dr[:, 2] / r)
    p0 = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="rad")

    # Velocity vector
    dr = (rsat + vsat) - (d.robs[d.mask] + d.vobs[d.mask])
    r = np.linalg.norm(dr, axis=1)
    ra = np.arctan2(dr[:, 1], dr[:, 0])
    dec = np.arcsin(dr[:, 2] / r)
    p1 = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="rad")

    # Offsets
    rxsat, rysat = p0.spherical_offsets_to(p1)
    rxobs, ryobs = p0.spherical_offsets_to(pobs)

    # Residuals
    alpha = -np.arctan2(rysat, rxsat)
    ca, sa = np.cos(alpha), np.sin(alpha)
    dt = (rxobs * ca - ryobs * sa) / np.sqrt(rxsat**2 + rysat**2)
    dr = rxobs * sa + ryobs * ca

    return dt, dr.to(u.deg).value


def format_time_for_output(t):
    fday = t.mjd - np.floor(t.mjd)
    day = t.datetime.day + fday
    year, month = t.datetime.year, t.datetime.month
    return f"{year:04d}{month:02d}{day:05.2f}"
