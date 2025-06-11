# Improved and annotated version of GW VT calculator


from __future__ import annotations

import multiprocessing as multi
from typing import Callable, Optional

import astropy.cosmology as cosmo
import astropy.units as u
import h5py
import lal
import lalsimulation as ls
import numpy as np
from tqdm import tqdm


# Constants and default parameters
DEFAULT_SENSITIVITY = ls.SimNoisePSDaLIGO175MpcT1800545
DEFAULT_WAVEFORM = ls.IMRPhenomPv2
PLANCK = cosmo.Planck15


def next_pow_two(x: int) -> int:
    return 2 ** int(np.ceil(np.log2(x)))


def optimal_snr(
    m1: float,
    m2: float,
    z: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[[lal.REAL8FrequencySeries, float], None]] = None,
    approximant: Optional[int] = None,
) -> float:
    psd_fn = psd_fn or DEFAULT_SENSITIVITY
    approximant = approximant or DEFAULT_WAVEFORM

    dL = PLANCK.luminosity_distance(z).to(u.Gpc).value
    m1z, m2z = m1 * (1 + z), m2 * (1 + z)

    tmax = (
        ls.SimInspiralChirpTimeBound(fmin, m1z * lal.MSUN_SI, m2z * lal.MSUN_SI, 0, 0)
        + 2.0
    )
    df = max(1.0 / next_pow_two(tmax), dfmin)
    fmax = 2048.0

    hp, _ = ls.SimInspiralChooseFDWaveform(
        m1z * lal.MSUN_SI,
        m2z * lal.MSUN_SI,
        *(0.0,) * 6,
        dL * 1e9 * lal.PC_SI,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        df,
        fmin,
        fmax,
        fref,
        None,
        approximant,
    )

    Nf = int(round(fmax / df)) + 1
    psd = lal.CreateREAL8FrequencySeries("psd", 0, 0.0, df, lal.DimensionlessUnit, Nf)
    psd_fn(psd, psdstart)

    return ls.MeasureSNRFD(hp, psd, psdstart, -1.0)


def fraction_above_threshold(
    m1,
    m2,
    z,
    snr_thresh,
    fmin=19.0,
    dfmin=0.0,
    fref=40.0,
    psdstart=20.0,
    psd_fn=None,
    approximant=None,
) -> float:
    if z == 0.0:
        return 1.0
    psd_fn = psd_fn or DEFAULT_SENSITIVITY

    rho_max = optimal_snr(
        m1,
        m2,
        z,
        fmin=fmin,
        dfmin=dfmin,
        fref=fref,
        psdstart=psdstart,
        psd_fn=psd_fn,
        approximant=approximant,
    )

    w = snr_thresh / rho_max
    if w > 1.0:
        return 0.0

    a2, a4, a8 = 0.374222, 2.04216, -2.63948
    return (
        a2 * (1 - w) ** 2
        + a4 * (1 - w) ** 4
        + a8 * (1 - w) ** 8
        + (1 - a2 - a4 - a8) * (1 - w) ** 10
    )


def pdet_from_mass_redshift(
    m1,
    m2,
    z,
    thresh,
    fmin=19.0,
    dfmin=0.0,
    fref=40.0,
    psdstart=20.0,
    psd_fn=None,
    approximant=None,
) -> float:
    psd_fn = psd_fn or DEFAULT_SENSITIVITY
    p_det = fraction_above_threshold(
        m1, m2, z, thresh, fmin, dfmin, fref, psdstart, psd_fn, approximant
    )
    return p_det


def _pdet_worker(args):
    return pdet_from_mass_redshift(*args)


def pdet_from_masses_redshifts(m1s, m2s, zs, thresh, psd_fn=None, processes=None):
    psd_fn = psd_fn or DEFAULT_SENSITIVITY
    processes = processes or multi.cpu_count()

    with multi.Pool(processes) as pool:
        args = [
            (m1, m2, z, thresh, 19.0, 0.0, 40.0, 20.0, psd_fn, DEFAULT_WAVEFORM)
            for m1, m2, z in zip(m1s, m2s, zs)
        ]
        vts = list(tqdm(pool.imap(_pdet_worker, args), total=len(zs)))
    return np.array(vts)


def main():
    output = "./pdet_with_Zuniform_injections.hdf5"
    samples = 100000

    m1s = np.random.uniform(0.5, 200.0, samples)
    m2s = np.random.uniform(0.5, 200.0, samples)
    zs = np.random.uniform(0.0, 4.0, samples)
    sampling_pdf = np.full((samples,), 1.0 / ((200.0 - 0.5) ** 2 * (4.0 - 0.0)))
    pdet = pdet_from_masses_redshifts(m1s, m2s, zs, 8.0)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = (
            "This file contains the pdet values for a set of "
            "mass and redshift injections. The masses are uniformly "
            "distributed between 0.5 and 200 solar masses, and the redshifts "
            "are uniformly distributed between 0 and 4. The pdet values are "
            "calculated using the IMRPhenomPv2 waveform approximant and the "
            "SimNoisePSDaLIGO175MpcT1800545 sensitivity curve."
        )
        f.attrs["waveform"] = "IMRPhenomPv2"
        f.attrs["sensitivity"] = "SimNoisePSDaLIGO175MpcT1800545"
        f.create_dataset("names", data=np.array(["mass_1_source", "mass_2_source", "redshift"]))
        f.create_dataset("mass_1_source", data=m1s)
        f.create_dataset("mass_2_source", data=m2s)
        f.create_dataset("redshift", data=zs)
        f.create_dataset("sampling_pdf", data=sampling_pdf)
        f.create_dataset("pdet", data=pdet)


if __name__ == "__main__":
    main()
