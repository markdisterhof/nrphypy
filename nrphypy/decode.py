'''
Decoders for sync-related data in 5G.
Copyright 2021 Mark Disterhof.
'''
from typing import Union
import numpy as np
import signals
import ssb


def decode_pss(received_data: np.ndarray) -> [int, int, int]:
    """TODO Refactor
    Find SSB with highest crosscorrelation to undistorted PSS available in resource grid 
    and extract N_ID_2, k_ssb and l_ssb

    Args:
        received_data ([complex, complex]): 2D resource grid in which to search for SSB

    Returns:
        tuple: N_ID_2, k_ssb, l_ssb
    """
    rec_pss_sym = np.array(received_data)
    nid2_candidates = np.array(
        [signals.pss(N_ID2=n_id2) for n_id2 in range(3)])

    corr = np.zeros(
        (3, received_data.shape[0], received_data.shape[1]), dtype=complex)

    for (i, pss_i) in enumerate(nid2_candidates):
        rgrid_mask = np.zeros(rec_pss_sym.shape, dtype=complex)
        rgrid_mask[:240, :4
                   ] += ssb.map_pss(pss_i)

        for l in range(received_data.shape[1]):
            for k in range(received_data.shape[0]):
                corr[i, k, l] = np.multiply(
                    np.roll(
                        np.roll(rgrid_mask, l, axis=1),
                        k, axis=0),
                    rec_pss_sym.real).sum()

    return np.unravel_index(np.argmax(corr, axis=None), corr.shape)


def pss_correlate(ofdm_sym: Union[np.ndarray, list]) -> [int, int, int]:
    """Correlate PSS on given data

    Args:
        ofdm_sym (Union[np.ndarray, list]): One OFDM Symbol

    Returns:
        [int, int, int]: NID_2, k_ssb, max_correlation
    """

    ofdm_sym = ofdm_sym.flatten()
    corr = np.array([np.correlate(np.real(ofdm_sym), signals.pss(nid2))
                    for nid2 in range(3)], dtype=float)
    (nid2, pss_start) = np.unravel_index(np.argmax(corr), corr.shape)
    return nid2, pss_start-56, corr[nid2, pss_start]/127.


def decode_sss(sss_data: Union[np.ndarray, list], nid2: int) -> int:
    """Extract N_ID_1 through crosscorrelation

    Args:
        sss_data (Union[np.ndarray, list]): Unmapped SSS
        nid2 (int): Cell ID sector

    Returns:
        int: NID_1
    """

    sss_candidates = np.array([signals.sss(
        N_ID1=nid1,
        N_ID2=nid2) for nid1 in range(336)])

    corr = np.zeros(len(sss_candidates), dtype=complex)

    for i, sss_i in enumerate(sss_candidates):
        corr[i] = np.abs(np.multiply(sss_i, sss_data).sum())

    return int(np.argmax(corr, axis=None))


def decode_pbch(pbch_data: Union[np.ndarray, list], L_max: int, N_ID_Cell: int, i_SSB: int) ->np.ndarray:
    """Extract PBCH payload bits from PBCH symbols

    Args:
        pbch_data (Union[np.ndarray, list]): Complex PBCH symbols
        L_max (int): Maximum number of candidate SS/PBCH blocks in a half frame
        N_ID_Cell (int): Cell ID
        i_SSB (int): SSB index per half frame

    Raises:
        ValueError: PBCH payload data must be 864 symbols

    Returns:
        np.ndarray: Descrambled PBCH bits 
    """    
    # get bits from complex symbols
    b = signals.inv_sym_qpsk(np.array(pbch_data, dtype=complex))
    # descramble that
    M_bit = len(b)
    if not M_bit == 864:
        raise ValueError('PBCH payload data must be 864 symbols')
    v = None
    if L_max == 4:
        v = i_SSB % 2**2
    else:
        v = i_SSB % 2**3

    c = signals.prsg((1+v) * M_bit, N_ID_Cell)
    scr = [c[i + v * M_bit] for i in range(M_bit)]

    return np.array([int(x) for x in np.logical_xor(b, scr)])
