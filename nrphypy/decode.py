'''
Decoders for sync-related data in 5G.
Copyright 2021 Mark Disterhof.
'''
from typing import Union
import numpy as np
import signals
import ssb


def _pss_candidates(N):
    pss_candidates = np.zeros((3, N), dtype=complex)
    pss_candidates[:, 56:56+127] = [signals.pss(i) for i in range(3)]
    pss_candidates_t = [np.fft.ifft(pss_candidates[i]) for i in range(3)]
    return pss_candidates_t


def sync_pss(received_data: np.ndarray, fft_size: int, threshold: float, pss_candidates: np.ndarray = None) -> np.ndarray:
    """TODO Process time domain samples to return array of 2d ssbs 

    Args:
        received_data (np.ndarray): [description]
        fft_size (int): [description]

    Returns:
        np.ndarray: [description]
    """
    # 3x 1d vector containing each ssb nid2s, offsets in rec_data and freq offsets
    nid2, epsilon, sample_offs = freq_time_sync(
        received_data=received_data, fft_size=fft_size, threshold=None, pss_candidates=pss_candidates)
    sample_cut = len(received_data)  # no of processed sample

    while not len(sample_offs) == 0 and  sample_offs[len(sample_offs)-1] + 4 * fft_size > len(received_data):
        # case received_data contains pss at the end but not rest of ssb
        # remove from found ssbs, sample_cut is no of processed samples (to be removed from memory)
        nid2 = nid2[:len(nid2)-1]
        sample_cut = sample_offs[len(sample_offs)-1]
        sample_offs = sample_offs[:len(sample_offs)-1]
        epsilon = epsilon[:len(epsilon)-1]  

    if len(sample_offs) == 0:
        return np.array([], dtype=complex), [], sample_cut, []

    ssbs_f = np.zeros(shape=(len(sample_offs), 4, fft_size), dtype=complex)
    for i, offs in enumerate(sample_offs):
        ssbs_f[i] = received_data[offs:offs+4*fft_size].reshape(4, fft_size)
    for i_ssb in range(len(ssbs_f)):
        for sym in range(4):
            ssbs_f[i_ssb, sym] *= np.exp(-1*epsilon[i_ssb]
                                         * 1j*2*np.pi*np.arange(fft_size)/fft_size)
            ssbs_f[i_ssb, sym] = np.fft.fft(ssbs_f[i_ssb, sym])
    gr = np.array([ssbs_f[i].T for i in range(len(ssbs_f))])[:, :240, :]
    return gr, nid2, sample_cut, sample_offs


def freq_time_sync(received_data: np.ndarray, fft_size: int, threshold: float, pss_candidates: np.ndarray = None) -> [int, int, int]:
    """TODO Conventional syncronization in time and frequency in time domain using 5g PSS as per:
    D. Wang et al.: Novel PSS Timing Synchronization Algorithm for Cell Search in 5G NR System
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9312170

    Args:
        received_data (np.ndarray): [description]

    Returns:
        [int, int, int]: NID2, epsilon, offset to ssb_carr0/samples

    """

    # prepare
    rec = np.array(received_data).flatten()
    if np.var(rec) == 0.0:
        return [],[],[]
    rec /= np.average(rec)
    # if pss_candidate provided do not compute iffts again etc.
    if pss_candidates == None:
        pss_candidates = _pss_candidates(fft_size)
    #
    '''
    B[t,n,k], where:
    t in {0,1,2}, NID2
    n in {0, len(rec)- len(pss_sequence)} cross-correlation sequence, which calculated starting from the n-th point of the receiving sequence
    k in {0,..,fft_size} fft point index
    '''

    mod_t = np.array([np.exp(k*1j*2*np.pi*np.arange(fft_size)/fft_size)
                     for k in range(fft_size)])  # modulator for integer freq shift
    B = np.zeros((len(pss_candidates), len(rec) -
                 fft_size + 1, fft_size), dtype=float)

    for t in range(len(pss_candidates)):  # for each NID2
        for k in range(len(B[0, 0])):  # for each carrier offset
            B[t, :, k] = np.abs(np.correlate(
                rec, pss_candidates[t] * mod_t[k]))
    C = np.array([[np.max(B[t, :, i]) for i in range(fft_size)]
                 for t in range(len(B))])
    if threshold == None:
        if np.average(B) <= 0.00001:
            # avoid avg^{Cell}(B)== 0 for next if case (div by zero)
            threshold = 1.
        elif np.max(B) / np.average(B)/1.1 < 10.:
            # case only noise, no meaningful threshold, threshold higher than
            threshold = np.max(B) + 1.
        else:
            threshold = np.max(B) - np.average(B)*1.1

    peaks = np.argwhere(B >= threshold)

    if len(peaks) > 100:
        raise Warning('threshold unreasonably low, too much memory is going to be allocated. threshold: {}, max(B): {}\ncontinuing with:{}'.format(
            threshold, np.max(B), np.max(B)*0.9))
        return freq_time_sync(rec, fft_size, np.max(B)*0.9, pss_candidates)

    ffo = np.angle(C[peaks[:, 0], peaks[:, 2]])/np.pi/2
    epsilon = peaks[:, 2] + ffo
    return [peaks[:, 0], epsilon, peaks[:, 1]]


def pss_correlate(received_data: np.ndarray, fft_size: int, pss_candidates: np.ndarray = None) -> int:
    
    # prepare
    rec = np.array(received_data).flatten()
    if np.var(rec) == 0.0:
        return [],[],[]
    rec /= np.average(rec)
    # if pss_candidate provided do not compute iffts again etc.
    if pss_candidates == None:
        pss_candidates = _pss_candidates(fft_size)
    #
    '''
    B[t,n,k], where:
    t in {0,1,2}, NID2
    n in {0, len(rec)- len(pss_sequence)} cross-correlation sequence, which calculated starting from the n-th point of the receiving sequence
    k in {0,..,fft_size} fft point index
    '''

    mod_t = np.array([np.exp(k*1j*2*np.pi*np.arange(fft_size)/fft_size)
                     for k in range(fft_size)])  # modulator for integer freq shift
    B = np.zeros((len(pss_candidates), len(rec) -
                 fft_size + 1, fft_size), dtype=float)

    for t in range(len(pss_candidates)):  # for each NID2
        for k in range(len(B[0, 0])):  # for each carrier offset
            B[t, :, k] = np.abs(np.correlate(
                rec, pss_candidates[t] * mod_t[k]))
    return np.unravel_index(np.argmax(B, axis=None), B.shape)[0]


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
        corr[i] = np.abs(np.correlate(sss_data,sss_i))
        
    return int(np.argmax(corr, axis=None))


def decode_pbch(pbch_data: Union[np.ndarray, list], L_max: int, N_ID_Cell: int, i_SSB: int) -> np.ndarray:
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

def dmrs(dmrs_data: Union[np.ndarray, list], N_ID_Cell:int, L_max:int) -> int:
    corr = np.array([np.correlate(dmrs_data, can_dmrs) for can_dmrs in [signals.dmrs(x,N_ID_Cell,L_max) for x in range(L_max)]])
    return np.argmax(np.abs(corr))


def dmrs_eq(ssb_data, i_ssb, N_ID_Cell, L_max, deg= 10):
    ssb_ = np.array(ssb_data)
    _, dmrs_data = ssb.unmap_pbch(ssb_,N_ID_Cell%4)
    can = signals.dmrs(i_ssb, N_ID_Cell, L_max)
    hest0 = dmrs_data/can
    dmrs_mask = ssb.map_pbch(np.zeros(432), hest0, N_ID_Cell%4)
    for sym in range(1,4):
        idx = np.where(dmrs_mask.T[sym] != 0)[0]
        x = idx
        y = hest0[:len(idx)]
        hest0 = hest0[len(idx):]
        p = np.poly1d(np.polyfit(x,y,deg))
        ssb_[:,sym] /=p(range(240))
        
    return ssb_