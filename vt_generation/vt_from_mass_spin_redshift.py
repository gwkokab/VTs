# Copyright 2023 The GWKokab Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import multiprocessing as multi
from typing import Callable, Optional

import astropy.cosmology as cosmo
import astropy.units as u
import lal
import lalsimulation as ls
import numpy as np
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm

# Constants
sensitivity = ls.SimNoisePSDaLIGO175MpcT1800545
waveform = ls.IMRPhenomPv2
planck = cosmo.Planck15


def next_pow_two(x: int) -> int:
    return 2 ** int(np.ceil(np.log2(x)))


def optimal_snr(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    psd_fn = psd_fn or sensitivity
    approximant = approximant or waveform

    dL = planck.luminosity_distance(z).to(u.Gpc).value
    m1z, m2z = m1 * (1 + z), m2 * (1 + z)

    tmax = ls.SimInspiralChirpTimeBound(fmin, m1z * lal.MSUN_SI, m2z * lal.MSUN_SI, a1z, a2z) + 2.0
    df = max(1.0 / next_pow_two(tmax), dfmin)
    fmax = 2048.0

    hp, _ = ls.SimInspiralChooseFDWaveform(
        m1z * lal.MSUN_SI, m2z * lal.MSUN_SI,
        0.0, 0.0, a1z, 0.0, 0.0, a2z,
        dL * 1e9 * lal.PC_SI,
        0.0, 0.0, 0.0, 0.0, 0.0,
        df, fmin, fmax, fref,
        None, approximant
    )

    Nf = int(round(fmax / df)) + 1
    psd = lal.CreateREAL8FrequencySeries("psd", 0, 0.0, df, lal.DimensionlessUnit, Nf)
    psd_fn(psd, psdstart)
    return ls.MeasureSNRFD(hp, psd, psdstart, -1.0)


def fraction_above_threshold(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    snr_thresh: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    if z == 0.0:
        return 1.0

    rho_max = optimal_snr(
        m1, m2, a1z, a2z, z,
        fmin=fmin, dfmin=dfmin, fref=fref,
        psdstart=psdstart, psd_fn=psd_fn, approximant=approximant
    )
    w = snr_thresh / rho_max
    if w > 1.0:
        return 0.0

    a2, a4, a8 = 0.374222, 2.04216, -2.63948
    return (
        a2 * (1 - w) ** 2 +
        a4 * (1 - w) ** 4 +
        a8 * (1 - w) ** 8 +
        (1 - a2 - a4 - a8) * (1 - w) ** 10
    )


def vt_from_mass_spin(
    m1: float, m2: float, a1z: float, a2z: float, z: float,
    analysis_time: float, thresh: float,
    fmin: float = 19.0, dfmin: float = 0.0, fref: float = 40.0, psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None, approximant: Optional[int] = None
) -> float:
    psd_fn = psd_fn or sensitivity

    p_det = fraction_above_threshold(
        m1, m2, a1z, a2z, z, thresh,
        fmin, dfmin, fref, psdstart, psd_fn, approximant
    )
    dVc_dz = planck.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value
    return 4 * np.pi * analysis_time * p_det * dVc_dz / (1 + z)


def _vt_worker(args):
    return vt_from_mass_spin(*args)


def vts_from_masses_spins(
    m1s, m2s, a1zs, a2zs, zs, analysis_time, thresh,
    psd_fn=None, processes=None
):
    psd_fn = psd_fn or sensitivity
    processes = processes or multi.cpu_count()

    args = [
        (m1, m2, a1, a2, z, analysis_time, thresh,
         19.0, 0.0, 40.0, 20.0, psd_fn, waveform)
        for m1, m2, a1, a2, z in zip(m1s, m2s, a1zs, a2zs, zs)
    ]

    with multi.Pool(processes) as pool:
        vts = list(tqdm(pool.imap(_vt_worker, args), total=len(zs)))

    return np.array(vts)


def redshift_model(N: int, zmax: float = 2.0, kappa: float = 2.7, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    z_grid = np.linspace(1e-3, zmax, 10000)
    dVc_dz = planck.differential_comoving_volume(z_grid).value
    pz = dVc_dz * (1 + z_grid) ** (kappa - 1)
    cdf = np.cumsum(pz)
    cdf /= cdf[-1]
    inv_cdf = interp1d(cdf, z_grid, bounds_error=False, fill_value=(z_grid[0], z_grid[-1]))
    return inv_cdf(np.random.uniform(0.0, 1.0, N))


def main():
    duration = 1.0 / 365.0  # 1 day
    samples = 10000000
    output = "./vt_with_Zuniform_injections_spins.hdf5"

    m1s = np.random.uniform(0.5, 200.0, samples)
    m2s = np.random.uniform(0.5, 200.0, samples)
    a1zs = np.random.uniform(-1, 1, samples)
    a2zs = np.random.uniform(-1, 1, samples)
    zs = np.random.uniform(0.0, 4.0, samples)

    vts = vts_from_masses_spins(m1s, m2s, a1zs, a2zs, zs, duration, 8.0)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = "VT estimation with aligned-spin model, 1-day observation"
        f.attrs["waveform"] = "IMRPhenomPv2"
        f.attrs["sensitivity"] = "aLIGO O4"
        f.create_dataset("mass_1_source", data=m1s)
        f.create_dataset("mass_2_source", data=m2s)
        f.create_dataset("a_1", data=a1zs)
        f.create_dataset("a_2", data=a2zs)
        f.create_dataset("redshift", data=zs)
        f.create_dataset("VT", data=vts)


if __name__ == "__main__":
    main()
