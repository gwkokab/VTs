#  Copyright 2024 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import multiprocessing as multi
from typing_extensions import Callable, Optional

import astropy.cosmology as cosmo
import astropy.units as u
import h5py
import lal
import lalsimulation as ls
import numpy as np
from jax import numpy as jnp


def next_pow_two(x: int) -> int:
    """
    Calculate the next power of two
    :param x: input number
    :return: next power of two
    """
    x2 = 1
    while x2 < x:
        x2 = x2 << 1
    return x2


# Define the PSD and waveform approximant
sensitivity = ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087
waveform = ls.IMRPhenomPv2


def optimal_snr(
    m1: float,
    m2: float,
    z: float,
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

    if psd_fn is None:
        psd_fn = sensitivity  # Default PSD

    if approximant is None:
        approximant = waveform
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html

    # Get dL, the luminosity distance in Gpc and tmax, the chirp time
    dL = cosmo.Planck15.luminosity_distance(z).to(u.Gpc).value

    tmax = (
        ls.SimInspiralChirpTimeBound(
            fmin,
            m1 * (1 + z) * lal.MSUN_SI,
            m2 * (1 + z) * lal.MSUN_SI,
            0.0,
            0.0,
        )
        + 2.0
    )

    df = max(1.0 / next_pow_two(tmax), dfmin)
    fmax = 2048.0  # Hz --- based on max freq of 5-5 inspiral

    # Compute the GW strain. g(t) = hp*Fps + hc*Fxs
    hp, hc = ls.SimInspiralChooseFDWaveform(
        ((1 + z) * m1 * lal.MSUN_SI),  # REAL8_const_m1
        ((1 + z) * m2 * lal.MSUN_SI),  # REAL8_const_m2
        0.0,  # REAL8_const_S1x
        0.0,  # REAL8_const_S1y
        0.0,  # REAL8_const_S1z
        0.0,  # REAL8_const_S2x
        0.0,  # REAL8_const_S2y
        0.0,  # REAL8_const_S2z
        dL * 1e9 * lal.PC_SI,  # REAL8_const_distance
        0.0,  # REAL8_const_inclination
        0.0,  # REAL8_const_phiRef
        0.0,  # REAL8_const_longAscNodes
        0.0,  # REAL8_const_eccentricity
        0.0,  # REAL8_const_meanPerAno
        df,  # REAL8_const_deltaF
        fmin,  # REAL8_const_f_min
        fmax,  # REAL8_const_f_max
        fref,  # REAL8_f_ref
        None,  # Dict_LALpars
        approximant,  # Approximant_const_approximant
    )

    Nf = int(round(fmax / df)) + 1
    fs = jnp.linspace(0, fmax, Nf)

    # PSD unit: 1/Hz
    # sffs is the frequency series of the PSD
    sffs = lal.CreateREAL8FrequencySeries(
        "psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0]
    )
    psd_fn(sffs, psdstart)
    return ls.MeasureSNRFD(hp, sffs, psdstart, -1.0)


# Computing p_{det}. This is the probability of detection of a signal
def fraction_above_threshold(
    m1: float,
    m2: float,
    z: float,
    snr_thresh: float = 8.0,
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
        p_{det}

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
    if z == 0.0:
        return 1.0

    if psd_fn is None:
        psd_fn = sensitivity

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
    # approximation for the detection from Richard O'Shaughnessy et al. 2010
    a2, a4, a8 = 0.374222, 2.04216, -2.63948
    w = snr_thresh / rho_max

    if w > 1.0:
        return 0.0  # no detection
    else:
        P_det = (
            a2 * ((1 - w) ** 2)
            + a4 * ((1 - w) ** 4)
            + a8 * ((1 - w) ** 8)
            + (1 - a2 - a4 - a8) * ((1 - w) ** 10)
        )
    return P_det  # detection


# Computing VT
def vt_from_mass(
    m1: float,
    m2: float,
    thresh: float = 8.0,
    analysis_time: float = 1.0,
    cal_factor: float = 1.0,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    zmax: float = 1.0,
    psd_fn: Optional[Callable[..., int]] = None,
    approximant: Optional[int] = None,
) -> float:
    r"""
    Volume time calculations from mass and spin
    :param m1: Source-frame mass 1
    :param m2: Source-frame mass 2
    :param thresh: The detection threshold in SNR
    :param analysis_time: The total detector-frame searched time in years
    :param cal_factor: A time calibration factor to apply to the result
    :param fmin: The starting frequency for waveform generation
    :param dfmin: The minimum frequency spacing
    :param fref: The reference frequency
    :param psdstart: The starting frequency for the PSD
    :param zmax: The maximum redshift
    :param psd_fn: Function giving the assumed single-detector PSD
    :param approximant: Approximant
    :return: The sensitive time-volume in comoving Gpc^3-yr (assuming analysis_time is given in years).
    """

    if psd_fn is None:
        psd_fn = sensitivity

    def integrand(z) -> float:
        if z == 0.0:
            return 0.0
        else:
            p_det = fraction_above_threshold(
                m1,
                m2,
                z,
                thresh,
                fmin=fmin,
                dfmin=dfmin,
                fref=fref,
                psdstart=psdstart,
                psd_fn=psd_fn,
                approximant=approximant,
            )
            return (
                4
                * jnp.pi
                * cosmo.Planck15.differential_comoving_volume(z)
                .to(u.Gpc**3 / u.sr)
                .value
                / (1 + z)
                * p_det
            )

    zmin = 0.001

    while zmax - zmin > 1e-3:
        zhalf = 0.5 * (zmax + zmin)
        fhalf = fraction_above_threshold(
            m1,
            m2,
            zhalf,
            thresh,
            fmin=fmin,
            dfmin=dfmin,
            fref=fref,
            psdstart=psdstart,
            psd_fn=psd_fn,
            approximant=approximant,
        )

        if fhalf > 0.0:
            zmin = zhalf
        else:
            zmax = zhalf
    # calculate the volume integral, and multiply by the analysis time and calibration factor
    # We may need better integration methods
    zs = np.linspace(0.0, zmax, 50)
    ys = np.array([integrand(z) for z in zs])
    vol_integral = np.trapz(ys, zs)
    vt = analysis_time * vol_integral * cal_factor
    print("time, vol_int, vt :", analysis_time, vol_integral, vt)
    return vt


def vts_from_masses(
    m1s,
    m2s,
    thresh=8.0,
    analysis_time=1.0 / 365,
    cal_factor=1.0,
    psd_fn=None,
    processes=None,
):
    """
    Compute the sensitive volume-time for a grid of masses.
    :param m1s: The first mass in the grid.
    :param m2s: The second mass in the grid.
    :param thresh: The detection threshold in SNR
    :param analysis_time: The total detector-frame searched time in years
    :param cal_factor: A time calibration factor to apply to the result.
    :param psd_fn: Function giving the assumed single-detector PSD
    :param processes: The number of processes to use in parallel.
    :return: The sensitive time-volume in comoving Gpc^3-yr (assuming analysis_time is given in years).
    """
    if psd_fn is None:
        psd_fn = sensitivity

    if processes is None:
        processes = multi.cpu_count()

    pool = multi.Pool(processes)

    vts = pool.starmap(
        vt_from_mass,
        [
            (
                m1,
                m2,
                thresh,
                analysis_time,
                cal_factor,
                19.0,
                0.0,
                40.0,
                20.0,
                1.0,
                psd_fn,
                waveform,
            )
            for m1, m2 in zip(m1s, m2s)
        ],
    )

    pool.close()
    pool.join()

    return np.array(vts)


def main():
    days = 1.0  # take it from user as input
    duration = days / 365.0  # convert days to years
    output = "./masses_vt.hdf5"

    with h5py.File(output, "w-") as f:
        # we can calculate the VT without using the grid
        masses = np.linspace(0.5, 200, 100)
        m1_grid, m2_grid = np.meshgrid(masses, masses)
        m1s, m2s = m1_grid.ravel(), m2_grid.ravel()
        vts = vts_from_masses(
            m1s,
            m2s,
            thresh=8.0,
            analysis_time=duration,
        )

        VT_grid = vts.reshape(m1_grid.shape)

        f.create_dataset("m1", data=m1_grid)
        f.create_dataset("m2", data=m2_grid)
        f.create_dataset("VT", data=VT_grid)


if __name__ == "__main__":
    main()
