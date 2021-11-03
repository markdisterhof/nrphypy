'''
Various methods for generation of sync-related data in 5G.
Copyright 2021 Mark Disterhof.
'''
import numpy as np
import signals
from typing import Union


def map_pss(data: Union[np.ndarray, list], beta: float = 1.0) -> np.ndarray:
    """Mapping of PSS within an SS/PBCH block as per TS 38 211 V16.2.0 (2020-07) 7.4.3.1.1

    Args:
        data (Union[np.ndarray, list]): PSS data
        beta (float, optional): [description]. PSS power allocation factor. Defaults to 1.0

    Raises:
        ValueError: ValueError: PSS data must be 127 symbols

    Returns:
        np.ndarray: 2D array, SSB with mapped PSS
    """

    if not len(data) == 127:
        raise ValueError("PSS data must be 127 symbols")
    mask = np.zeros((240, 4), dtype=complex)
    data = np.array(data) * beta
    mask[56:183, 0] = data
    return mask


def map_sss(data: Union[np.ndarray, list], beta: float = 1.0) -> np.ndarray:
    """Mapping of SSS within an SS/PBCH block as per TS 38 211 V16.2.0 (2020-07) 7.4.3.1.2

    Args:
        data (Union[np.ndarray, list]): SSS data
        beta (float, optional): SSS power allocation factor. Defaults to 1.0

    Raises:
        ValueError: SSS data must be 127 symbols

    Returns:
        np.ndarray: 2D array, SSB with mapped SSS
    """

    if not len(data) == 127:
        raise ValueError("sss data must be 127 symbols")
    mask = np.zeros((240, 4), dtype=complex)
    data = np.array(data) * beta
    mask[56:183, 2] = data
    return mask


def map_pbch(data_pbch: Union[np.ndarray, list], data_dmrs: Union[np.ndarray, list], nu: int, beta_pbch: float = 1.0, beta_dmrs: float = 1.0) -> np.ndarray:
    """Mapping of PBCH and DM-RS within an SS/PBCH block as per TS 38 211 V16.2.0 (2020-07) 7.4.3.1.3

    Args:
        data_pbch (Union[np.ndarray, list]): PBCH data
        data_dmrs (Union[np.ndarray, list]): DM-RS data
        nu (int) : nu, as defined: N_ID_CELL % 4
        beta_pbch (float, optional): PBCH power allocation factor. Defaults to 1.0
        beta_dmrs (float, optional): DM-RS power allocation factor. Defaults to 1.0

    Raises:
        ValueError: data_pbch must be 432 symbols, data_dmrs must be 144 symbols

    Returns:
        np.ndarray: 2D array, SSB with mapped PBCH and DM-RS
    """
    if not len(data_pbch) == 432 or not len(data_dmrs) == 144:
        raise ValueError(
            "pbch is always 432 symbols, dmrs is always 144 symbols")
    i_dmrs, i_pbch = 0, 0

    data_pbch = np.array(data_pbch, dtype=complex) * beta_pbch
    data_dmrs = np.array(data_dmrs, dtype=complex) * beta_dmrs
    mask = np.zeros((240, 4), dtype=complex)

    for l in range(1, 4):
        k_range = range(240)
        if l == 2:
            k_range = np.concatenate((range(48), range(192, 240)))
        for k in k_range:
            if k % 4 == nu:
                mask[k, l] = data_dmrs[i_dmrs]
                i_dmrs += 1
            else:
                mask[k, l] = data_pbch[i_pbch]
                i_pbch += 1
    return mask


def ssb(N_ID1: int, N_ID2: int, L_max: int, i_ssb: int, pbch_data: Union[np.ndarray, list]) -> np.ndarray:
    """Generate SS/PBCH block 

    Args:
        N_ID1 (int): Cell ID group
        N_ID2 (int): Cell ID sector
        L_max (int): Maximum number of SS/PBCH blocks in a half frame
        i_ssb (int): Candidate SS/PBCH block index
        pbch_data (Union[np.ndarray, list]): Binary PBCH data

    Returns:
        np.ndarray: 2D array, SS/PBCH block 
    """
    N_ID_Cell = 3 * N_ID1 + N_ID2
    nu = N_ID_Cell % 4

    ssb = np.zeros((240, 4), dtype=complex)

    # pss mapping
    data_pss = signals.pss(N_ID2=N_ID2)
    ssb += map_pss(data_pss)

    # sss mapping
    data_sss = signals.sss(N_ID1=N_ID1, N_ID2=N_ID2)
    ssb += map_sss(data_sss)

    # pbch and dmrs mapping
    pbch = signals.pbch(pbch_data, i_ssb, N_ID_Cell, L_max)
    dmrs = signals.dmrs(i_ssb, N_ID_Cell, L_max, 0)
    ssb += map_pbch(pbch, dmrs, nu)

    return ssb


def unmap_pss(received_data: np.ndarray) -> np.ndarray:
    """Unmap PSS from given SSB

    Args:
        received_data (np.ndarray): SSB with shape (240,4)

    Raises:
        IndexError: Provided SSB is not (240,4) in shape

    Returns:
        np.ndarray: PSS data
    """

    if not received_data.shape == (240, 4):
        raise IndexError(
            'Provided SSB is not (240,4) in shape: {}'.format(received_data.shape))

    mask = map_pss(np.ones(127))

    return np.ma.masked_array(
        received_data.flatten(order='F'),
        np.logical_not(
            mask.flatten(order='F')
        )
    ).compressed()


def unmap_sss(received_data: np.ndarray) -> np.ndarray:
    """Unmap SSS from given SSB

    Args:
        received_data (np.ndarray): SSB with shape (240,4)

    Raises:
        IndexError: Provided SSB is not (240,4) in shape

    Returns:
        np.ndarray: PSS data
    """
    if not received_data.shape == (240, 4):
        raise IndexError(
            'Provided SSB is not (240,4) in shape: {}'.format(received_data.shape))

    mask = map_sss(np.ones(127))

    return np.ma.masked_array(
        received_data.flatten(order='F'),
        np.logical_not(
            mask.flatten(order='F')
        )
    ).compressed()


def unmap_pbch(received_data: np.ndarray, nu: int) -> (np.ndarray, np.ndarray):
    """Unmap PBCH and DM-RS from given resource grid

    Args:
        received_data (np.ndarray): SSB with shape (240,4)
        nu (int): nu, as defined: N_ID_CELL % 4

    Raises:
        IndexError: Provided SSB is not (240,4) in shape

    Returns:
        (np.ndarray, np.ndarray): PBCH and DM-RS data
    """
    if not received_data.shape == (240, 4):
        raise IndexError(
            'Provided SSB is not (240,4) in shape: {}'.format(received_data.shape))

    mask_pbch = map_pbch(np.ones(432), np.zeros(144), nu)
    mask_dmrs = map_pbch(np.zeros(432), np.ones(144), nu)

    data_pbch = np.ma.masked_array(
        received_data.flatten(order='F'),
        np.logical_not(
            mask_pbch.flatten(order='F')
        )
    ).compressed()

    data_dmrs = np.ma.masked_array(
        received_data.flatten(order='F'),
        np.logical_not(
            mask_dmrs.flatten(order='F')
        )
    ).compressed()

    return data_pbch, data_dmrs


def unmap_ssb(res_grid: np.ndarray, k_offs: int, l_offs: int) -> np.ndarray:
    """Recover SSB from provided resource grid with offsets

    Args:
        res_grid (np.ndarray): 2D resource grid
        ssb (np.ndarray): 2D SSB
        k_offs (int): Offset counted in multiples of Subcarriers
        l_offs (int): Offset counted in symbols
    Raises:
        IndexError: Provided res_grid is too small to hold an SSB

    Returns:
        np.ndarray: 2D resource grid with placed SSB
    """
    if res_grid.shape < (240+k_offs, 4+l_offs):
        raise IndexError(
            'Provided res_grid is too small to hold an SSB: {}'.format(res_grid.shape))

    return res_grid[k_offs:k_offs+240, l_offs:l_offs+4]


def map_ssb(res_grid: np.ndarray, ssb: np.ndarray, k_offs: int, l_offs: int) -> np.ndarray:
    """Place SSB in provided resource grid with offsets

    Args:
        res_grid (np.ndarray): 2D resource grid
        ssb (np.ndarray): 2D SSB
        k_offs (int): Offset counted in multiples of Subcarriers
        l_offs (int): Offset counted in symbols
    Raises:
        IndexError: Provided res_grid is too small to hold an SSB

    Returns:
        np.ndarray: 2D resource grid with placed SSB
    """
    if res_grid.shape < (240+k_offs, 4+l_offs):
        raise IndexError(
            'Provided res_grid is too small to hold an SSB: {}'.format(res_grid.shape))

    res_grid[k_offs:k_offs+240, l_offs:l_offs+4] = ssb
    return res_grid


def grid(n_carr: int = 240, N_ID1: int = 0, N_ID2: int = 0, k_ssb: int = 0, mu: int = 0, f: int = 0, shared_spectr: bool = False, paired_spectr: bool = False, pbch: Union[np.ndarray, list] = np.random.randint(2, size=864)) -> np.ndarray:
    """Produce a NR sync resource grid. 

    Args:
        n_carr (int, optional): Number of carriers. Defaults to 240.
        N_ID1 (int, optional): NID1. Defaults to 0.
        N_ID2 (int, optional): NID2. Defaults to 0.
        k_ssb (int, optional): Carrier offset from OffsetToPointA to SSB. Defaults to 0.
        mu (int, optional): Numerology index. Defaults to 0.
        f (int, optional): Operating frequency. Defaults to 0.
        shared_spectr (bool, optional): Shared spectrum use. Defaults to False.
        paired_spectr (bool, optional): Paired spectrum use. Defaults to False.
        pbch (Union[np.ndarray, list], optional): PBCH Payload bits. Defaults to np.random.randint(2, size=864).

    Raises:
        ValueError: n_carr is too small to fit resource grid

    Returns:
        np.ndarray: 2D Array, sync resource grid
    """
    if n_carr < 240 + k_ssb:
        raise ValueError(
            'Provided n_carr is too small. n_carr: {0}, min needed: {1}'.format(n_carr, 240+k_ssb))

    # gen rgrid for sync
    grid = get_sync_resource_grid_pbch(
        N_RB=n_carr//12, N_ID1=N_ID1, N_ID2=N_ID2, k_ssb=k_ssb, mu=mu, f=f, shared_spectr=shared_spectr, paired_spectr=paired_spectr, pbch_data=pbch)

    # fit grid with N_RB*12 carr into n_carr
    mask = np.zeros((n_carr, len(grid[0])), dtype=complex)
    mask[:len(grid), :len(grid[0])] = grid

    return mask


def get_sync_resource_grid_pbch(N_RB: int, N_ID1: int, N_ID2: int, k_ssb: int, mu: int, f: int, shared_spectr: bool, paired_spectr: bool, pbch_data: Union[np.ndarray, list]) -> np.ndarray:
    """Produce a NR sync resource grid. 

    Args:
        N_RB (int): Number of resourceblocks to use (N_RB * 12 = N_carriers). 
        N_ID1 (int): NID1. 
        N_ID2 (int): NID2. 
        k_ssb (int): Carrier offset from OffsetToPointA to SSB. 
        mu (int): Numerology index.
        f (int): Operating frequency.
        shared_spectr (bool): Shared spectrum use.
        paired_spectr (bool): Paired spectrum use.
        pbch (Union[np.ndarray, list]): PBCH Payload bits.

    Returns:
        np.ndarray: 2D Array, sync resource grid
    """

    N_ID_Cell = 3 * N_ID1 + N_ID2
    nu = N_ID_Cell % 4

    N_SC, N_SYMB = get_rgrid_dimensions(mu, N_RB)

    can_ids = get_ssb_candidate_idx(mu, f, shared_spectr, paired_spectr)
    ids = get_ssb_ids(can_ids, mu, shared_spectr)
    L_max = len(ids)
    res_grid = np.zeros(shape=(N_SC, N_SYMB), dtype=complex)

    pbch_data = np.array(pbch_data)
    pbch_data.resize(L_max*864)
    pbch_data_arr = pbch_data.reshape(L_max, 864)

    for i_ssb, idx in enumerate(ids):
        ssb_i = ssb(N_ID1=N_ID1, N_ID2=N_ID2, L_max=L_max,
                    i_ssb=i_ssb, pbch_data=pbch_data_arr[i_ssb])
        res_grid = map_ssb(res_grid, ssb_i, k_ssb, idx)
    return res_grid


def get_rgrid_dimensions(mu: int, n_rb: int) -> [int, int]:
    """Returns number of subcarriers and number of symbols in a frame. 

    See TS 38.211 Table 4.3.2-1,
        TS 38.101-1 Table 5.3.2-1

    Args:
        mu (int): Numerology index {0,1,2,3,4}
        n_rb (int): Number of resource blocks

    Returns:
        (int, int): N_SC, N_SYMB_FRAME
    """
    N_SC_RB = 12
    N_SC = n_rb * N_SC_RB

    N_SYMB_SLOT = 14
    N_SLOTS_FRAME = 2 ** mu * 10
    N_SYMB_FRAME = N_SYMB_SLOT * N_SLOTS_FRAME

    return N_SC, N_SYMB_FRAME


def get_ssb_ids(candidate_ids: Union[np.ndarray, list], mu: int, shared_spectr: bool) -> Union[np.ndarray, list]:
    """Compute indexes of the first symbols of the the cadidate SS/PBCH blocks as per 38.213 V16.3.0 (2020-11) 4.1 Cell search

    Args:
        candidate_ids (Union[np.ndarray, list]): cadidate indices for SS/PBCH blocks
        mu (int): Numerology index
        shared_spectr (bool): shared spectrum channel access

    Returns:
        Union[np.ndarray, list]: List of SSB indices
    """
    if shared_spectr:
        if (len(candidate_ids) == 10 and mu == 0) or (len(candidate_ids) == 20 and mu == 1):
            return candidate_ids[:8]

    return candidate_ids


def get_ssb_candidate_idx(mu: int, f: int, ssca: bool, paired: bool) -> Union[np.ndarray, list]:
    """Compute indexes of the first symbols of the the cadidate SS/PBCH blocks as per 38.213 V16.3.0 (2020-11) 4.1 Cell search

    Args:
        mu (int): Numerology index
        f (int): Operating frequency
        ssca (bool): Shared spectrum channel access.
        paired (bool): Paired spectrum operation. 

    Returns:
        Union[np.ndarray, list]: List of cadidate SSB indices 
    """
    n = []
    i = []
    scs = int(2**mu * 15e3)

    if scs == 15e3:
        # case A
        i = np.array([2, 8])
        if ssca:
            # shared spectrum channel access, as described in [15, TS 37.213]s
            n = np.array([0, 1, 2, 3, 4])
        elif f <= 3e9:
            n = np.array([0, 1])
        else:
            n = np.array([0, 1, 2, 3])
        n *= 14

    elif scs == 30e3 and not (ssca or paired):
        # case B
        i = np.array([4, 8, 16, 20])
        if f <= 3e9:
            n = np.array([0])
        else:
            n = np.array([0, 1])
        n *= 28

    elif scs == 30e3 and (ssca or paired):
        # case C
        i = np.array([2, 8])
        if not ssca:
            # shared spectrum channel access, as described in [15, TS 37.213]s
            if paired:
                if f <= 3e9:
                    n = np.array([0, 1])
                else:
                    n = np.array([0, 1, 2, 3])
            else:
                if f <= 1.88e9:
                    n = np.array([0, 1])
                else:
                    n = np.array([0, 1, 2, 3])
        else:
            n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        n *= 14
    elif scs == 120e3:
        # case D
        if f >= 6e9:
            i = np.array([4, 8, 16, 20])
            n = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10,
                         11, 12, 13, 15, 16, 17, 18])
            n *= 28
    elif scs == 240e3:
        # case E
        if f >= 6e9:
            i = np.array([8, 12, 16, 20, 32, 36, 40, 44])
            n = np.array([0, 1, 2, 3, 5, 6, 7, 8])
            n *= 56
    can_ids = np.array([[a + b for a in i] for b in n], dtype=int).flatten()
    if can_ids is None:
        can_ids = np.array([], dtype=int)
    return can_ids
