# Copyright 2023 The GWKokab Authors
# Licensed under the Apache License, Version 2.0

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


# Constants and configuration
sensitivity = ls.SimNoisePSDaLIGO175MpcT1800545  # Power spectral density model
waveform = ls.TaylorF2Ecc  # Frequency-domain waveform model with eccentricity
planck = cosmo.Planck15  # Cosmology for computing luminosity distances


def next_pow_two(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    return 2 ** int(np.ceil(np.log2(x)))


def optimal_snr(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    ecc: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    r"""
    Return the optimal SNR of a signal.
    :param m1: The source-frame mass 1.
    :param m2: The source-frame mass 2.
    :param z: The redshift
    :param fmin: The starting frequency for waveform generation.
    :param dfmin:
    :param fref:
    :param psdstart:
    :param psd_fn: A function that returns the detector PSD at a given frequency, you can choose any given in lalsimulation.
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html
    :param approximant:
    :return: The SNR of a face-on, overhead source.
    """
    psd_fn = psd_fn or sensitivity
    approximant = approximant or waveform
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html

    # Redshifted masses and distance
    dL = planck.luminosity_distance(z).to(u.Gpc).value  # Luminosity distance in Gpc
    m1z, m2z = m1 * (1 + z), m2 * (1 + z)

    # Determine frequency resolution from estimated waveform duration
    tmax = (
        ls.SimInspiralChirpTimeBound(
            fmin, m1z * lal.MSUN_SI, m2z * lal.MSUN_SI, a1z, a2z
        )
        + 2.0
    )
    df = max(1.0 / next_pow_two(tmax), dfmin)
    fmax = 2048.0  # Nyquist frequency # Hz --- based on max freq of 5-5 inspiral

    # Generate the frequency-domain waveform
    hp, _ = ls.SimInspiralChooseFDWaveform(
        m1z * lal.MSUN_SI,
        m2z * lal.MSUN_SI,  # source-frame masses in SI units
        0.0,
        0.0,
        a1z,
        0.0,
        0.0,
        a2z,  # spins in dimensionless units
        dL * 1e9 * lal.PC_SI,  # distance in SI units (Gpc to m)
        0.0,
        0.0,
        0.0,  # inclination, polarization, and phase
        ecc,
        0.0,  # eccentricity and mean anomaly
        df,
        fmin,
        fmax,
        fref,  # frequency resolution, min and max frequencies, reference frequency
        None,
        approximant,  # waveform approximant
    )

    # Construct PSD and compute matched-filter SNR
    Nf = int(round(fmax / df)) + 1
    psd = lal.CreateREAL8FrequencySeries(
        "psd", 0, 0.0, df, lal.DimensionlessUnit, Nf
    )  # Create frequency series for PSD, unit 1/Hz
    psd_fn(psd, psdstart)

    return ls.MeasureSNRFD(hp, psd, psdstart, -1.0)  # Compute matched-filter SNR


# Computing p_{det}. This is the probability of detection of a signal
def fraction_above_threshold(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    ecc: float,
    snr_thresh: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    r"""
    Computes
    .. math::
        p_{det} using analytical approximation.

    :param m1: Source-frame mass 1.
    :param m2: Source-frame mass 2.
    :param z: Redshift
    :param snr_thresh: The detection threshold in SNR
    :param fmin: The starting frequency for waveform generation
    :param dfmin: The minimum frequency spacing
    :param fref: The reference frequency
    :param psdstart: The starting frequency for the PSD
    :param psd_fn: Function giving the assumed single-detector PSD
    :param approximant: Approximant
    :return: Fraction of time above threshold
    """
    if z == 0.0:  # overhead source
        return 1.0  # If redshift is zero, the source is at the observer's frame, so detection probability is 1.0

    rho_max = optimal_snr(
        m1, m2, a1z, a2z, z, ecc, fmin, dfmin, fref, psdstart, psd_fn, approximant
    )
    w = snr_thresh / rho_max
    if w > 1.0:
        return 0.0  # If the threshold is greater than the maximum SNR, return 0, no detection.

    # Detection probability polynomial fit coefficients
    # approximation for the detection from Richard O'Shaughnessy et al. 2010
    else:
        a2, a4, a8 = 0.374222, 2.04216, -2.63948
        Pdet = (
            a2 * (1 - w) ** 2
            + a4 * (1 - w) ** 4
            + a8 * (1 - w) ** 8
            + (1 - a2 - a4 - a8) * (1 - w) ** 10
        )
    return Pdet


def pdet_from_mass_spin(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    ecc: float,
    z: float,
    thresh: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    """
    Wrapper function to compute the detection probability for one binary.
    """
    return fraction_above_threshold(
        m1,
        m2,
        a1z,
        a2z,
        z,
        ecc,
        thresh,
        fmin,
        dfmin,
        fref,
        psdstart,
        psd_fn,
        approximant,
    )


def _pdet_worker(args):
    """Helper function for multiprocessing map."""
    return pdet_from_mass_spin(*args)


def pdets_from_masses_spins(
    m1s, m2s, a1zs, a2zs, eccs, zs, thresh, psd_fn=None, processes=None
):
    """
    Evaluate the detection probability for a population of binary systems.
    """
    psd_fn = psd_fn or sensitivity
    processes = processes or multi.cpu_count()

    # Build argument list for parallel processing
    args = [
        (m1, m2, a1, a2, ecc, z, thresh, 19.0, 0.0, 40.0, 20.0, psd_fn, waveform)
        for m1, m2, a1, a2, ecc, z in zip(m1s, m2s, a1zs, a2zs, eccs, zs)
    ]

    with multi.Pool(processes) as pool:
        vts = list(tqdm(pool.imap(_pdet_worker, args), total=len(zs)))

    return np.array(vts)


def main():
    samples = 200000
    output = "./pdet_with_TaylorF2Ecc_uniform_injections.hdf5"

    params = {
        "mass_1_source_min": 0.5,
        "mass_1_source_max": 100.0,
        "mass_2_source_min": 0.5,
        "mass_2_source_max": 100.0,
        "a_1_min": 0.0,
        "a_1_max": 1.0,
        "a_2_min": 0.0,
        "a_2_max": 1.0,
        "eccentricity_min": 0.0,
        "eccentricity_max": 0.3,
        "redshift_min": 0.0,
        "redshift_max": 2.3,
    }

    m1s = np.random.uniform(
        params["mass_1_source_min"], params["mass_1_source_max"], samples
    )
    m2s = np.random.uniform(
        params["mass_2_source_min"], params["mass_2_source_max"], samples
    )
    a1zs = np.random.uniform(params["a_1_min"], params["a_1_max"], samples)
    a2zs = np.random.uniform(params["a_2_min"], params["a_2_max"], samples)
    eccs = np.random.uniform(
        params["eccentricity_min"], params["eccentricity_max"], samples
    )
    zs = np.random.uniform(params["redshift_min"], params["redshift_max"], samples)

    volume = (
        (params["mass_1_source_max"] - params["mass_1_source_min"])
        * (params["mass_2_source_max"] - params["mass_2_source_min"])
        * (params["a_1_max"] - params["a_1_min"])
        * (params["a_2_max"] - params["a_2_min"])
        * (params["eccentricity_max"] - params["eccentricity_min"])
        * (params["redshift_max"] - params["redshift_min"])
    )  # uniform priors in all params
    sampling_pdf = np.full(samples, 1.0 / volume)

    pdets = pdets_from_masses_spins(m1s, m2s, a1zs, a2zs, eccs, zs, 8.0)

    with h5py.File(output, "w") as f:
        for key, value in params.items():
            f.attrs[key] = value
        f.attrs["description"] = (
            "This file contains the pdet values for a population of binary systems "
            "with uniform priors in mass, spin, eccentricity, and redshift. "
            "The pdet values are computed using the TaylorF2Ecc waveform model "
            "and the SimNoisePSDaLIGO175MpcT1800545 sensitivity model. "
            f"The redshift is uniformly sampled from 0 to {params['redshift_max']}."
        )

        f.attrs["waveform"] = "TaylorF2Ecc"
        f.attrs["sensitivity"] = "SimNoisePSDaLIGO175MpcT1800545"
        f.attrs["redshift_model"] = f"Uniform in z from 0 to {params['redshift_max']}"
        f.create_dataset(
            "names",
            data=np.array(
                [
                    "mass_1_source",
                    "mass_2_source",
                    "a_1",
                    "a_2",
                    "eccentricity",
                    "redshift",
                ],
                dtype="S",
            ),
        )
        f.create_dataset("mass_1_source", data=m1s)
        f.create_dataset("mass_2_source", data=m2s)
        f.create_dataset("a_1", data=a1zs)
        f.create_dataset("a_2", data=a2zs)
        f.create_dataset("eccentricity", data=eccs)
        f.create_dataset("redshift", data=zs)
        f.create_dataset("sampling_pdf", data=sampling_pdf)
        f.create_dataset("pdet", data=pdets)


if __name__ == "__main__":
    main()
