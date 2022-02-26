'''
Methods for generation of sync signals in 5G.
Copyright 2021 Mark Disterhof.
'''
import numpy as np
from typing import Union


def prsg(M_PN: int, c_init: int) -> np.ndarray:
    """Compute c as per:
    TS 38.211 16.2.0 (2020-07) 5.2.1 Pseudo-random sequence generation

    Args:
        M_PN (int): Sequence length
        c_init (int): initializer

    Returns:
        np.ndarray: Pseudo-random sequence of length M_PN
    """
    c = np.zeros(M_PN)
    N_c = 1600
    x_1 = np.array([1], dtype=int)
    x_1.resize(len(c) + N_c)
    x_2 = np.zeros((len(c) + N_c))
    x_2_c = np.flip([int(x) for x in bin(c_init)[2:]]).copy()
    x_2_c.resize(30)
    x_2[:len(x_2_c)] = x_2_c

    for n in range(len(x_1)-31):
        x_1[n+31] = (x_1[n+3] + x_1[n]) % 2
        x_2[n+31] = (x_2[n+3] + x_2[n+2] + x_2[n+1] + x_2[n]) % 2

    for n in range(M_PN):
        c[n] = (x_1[n + N_c] + x_2[n + N_c]) % 2
    return c


def pss(N_ID2: int) -> np.ndarray:
    """Generate primary synchronization signal as per:
    TS 38 211 V16.2.0 (2020-07) 7.4.2.2 Primary synchronization signal

    Args:
        N_ID2 (int):  Cell ID sector

    Returns:
        np.ndarray: Primary synchronization signal sequence (127 symbols in {-1,1})
    """

    d_pss = np.zeros(127, dtype=int)
    x = np.resize([0, 1, 1, 0, 1, 1, 1], 127)

    for i in range(len(x)-7):
        x[i+7] = (x[i+4] + x[i]) % 2

    for n in range(len(d_pss)):
        d_pss[n] = 1-2*x[(n+43*N_ID2) % 127]

    return d_pss


def sss(N_ID1: int, N_ID2: int) -> np.ndarray:
    """Generate secondary synchronization signal as per:
    TS 138 211 V16.2.0 (2020-07) 7.4.2.3 Secondary synchronization signal

    Args:
        N_ID1 (int): Cell ID group
        N_ID2 (int): Cell ID sector

Returns:
        np.ndarray: Secondary synchronization signal sequence (127 symbols in {-1,1})
    """
    x_0 = np.resize([1, 0, 0, 0, 0, 0, 0], 127)
    x_1 = np.resize([1, 0, 0, 0, 0, 0, 0], 127)
    m_0 = 15*(N_ID1//112) + 5 * N_ID2
    m_1 = N_ID1 % 112

    for i in range(len(x_0)-8):
        x_0[i+7] = (x_0[i+4] + x_0[i]) % 2
        x_1[i+7] = (x_1[i+1] + x_1[i]) % 2

    d_sss = np.zeros(127, dtype=int)
    for n in range(len(d_sss)):
        d_sss[n] = (1-2*x_0[(n+m_0) % 127])*(1-2*x_1[(n+m_1) % 127])

    return d_sss


def dmrs(i_ssb: int, N_ID_Cell: int, L_max: int, n_hf: bool = False) -> np.ndarray:
    """Generate Demodulation reference signals for PBCH as per 
    TS 38.211 V16.2.0 (2020-07) 7.4.1.4 Demodulation reference signals for PBCH 

    Args:
        i_ssb (int): SS/PBCH block index
        N_ID_Cell (int): Cell identity ID
        L_max (int): Maximum number of SS/PBCH blocks in a half frame
        n_hf (bool, optional): Number of the half-frame in which the PBCH is transmitted in a frame n_hf=False: first half-frame,n_hf=True: second half-frame . Defaults to False.

    Returns:
        np.ndarray: Demodulation reference signals for PBCH
    """
    M = 144
    i_ssb_ = None

    if L_max == 4:
        i_ssb_ = i_ssb % 4 + 4 * int(n_hf)
    elif L_max > 4:
        i_ssb_ = i_ssb % 8

    c_init = int(2**11 * (i_ssb_ + 1) * ((N_ID_Cell // 4)+1) +
                 2**6 * (i_ssb_ + 1) +
                 (N_ID_Cell % 4))

    c = prsg(2 * M + 1, c_init)

    r = np.zeros(M, dtype=complex)
    for m in range(M):
        r[m] = (1 - 2 * c[2 * m] + (1 - 2 * c[2 * m + 1]) * 1j) / np.sqrt(2)  # compute complex symbols
    return r


def pbch(b: Union[np.ndarray, list], i_SSB: int, N_ID_Cell: int, L_max: int) -> np.ndarray:
    """Generate physical broadcast channel sequence as per:
    TS 38.211 V16.2.0 (2020-07) 7.3.3 Physical broadcast channel

    Args:
        b (Union[np.ndarray, list]): PBCH payload bits
        i_SSB (int): SS/PBCH block index
        N_ID_Cell (int): Cell identity ID
        L_max (int): Maximum number of SS/PBCH blocks in a half frame

    Raises:
        ValueError: PBCH payload data must be 864 symbols

    Returns:
        np.ndarray: Scrambled and QPSK modulated PBCH symbols
    """

    M_bit = len(b)
    if not M_bit == 864:
        raise ValueError('PBCH payload data must be 864 symbols')

    v = None
    if L_max == 4:
        v = i_SSB % 2**2
    else:
        v = i_SSB % 2**3

    c = prsg((1+v) * M_bit, N_ID_Cell)

    b_ = [(b[i] + c[i + v * M_bit]) % 2 for i in range(M_bit)]

    d_PBCH = sym_qpsk(b_)
    return d_PBCH


def sym_qpsk(b: Union[np.ndarray, list]) -> np.ndarray:
    """Modulate a list of bits with QPSK as per

    TS 38.211 V16.2.0 (2020-07) 5.1.3

    Args:
        b (Union[np.ndarray, list]): List of bits to QPSK-modulate

    Returns:
        np.ndarray: List of bits
    """

    return np.array(
        [(1-2*b[2*i]+1j*(1-2*b[2*i+1]))/np.sqrt(2) for i in range(len(b)//2)],
        dtype=complex)


def inv_sym_qpsk(c: Union[np.ndarray, list]) -> np.ndarray:
    """Demodulate a list of bits with QPSK as per

    TS 38.211 V16.2.0 (2020-07) 5.1.3

    Args:
        b (Union[np.ndarray, list]): List of QPSK-symbols to demodulate
    Returns:
        np.ndarray:
    """

    c = np.array(c, dtype=complex).flatten()  # weird python type error in gr
    if not len(c) % 2 == 0:
        c.resize(len(c)+1)
    b_ = np.array(
        [[int(np.round(np.real(i)*np.sqrt(2))), int(np.round(np.imag(i)*np.sqrt(2)))] for i in c], dtype=int
    ).flatten()
    return np.array([(1-b_i)//2 for b_i in b_], dtype=int)
