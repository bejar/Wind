"""
.. module:: Entropy

Entropy
*************

:Description: Entropy

    

:Authors: bejar
    

:Version: 

:Created on: 26/07/2019 9:57 

"""

from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
import numpy as np

def _embed(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.

    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    sf : float
        Sampling frequency
    method : str
        Spectral estimation method ::

        'fft' : Fourier Transform (via scipy.signal.periodogram)
        'welch' : Welch periodogram (via scipy.signal.welch)

    nperseg : str or int
        Length of each FFT segment for Welch method.
        If None, uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectral Density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} PSD(f) log_2[PSD(f)]

    Where :math:`PSD` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
       use of the entropy of the power spectrum. Electroencephalography
       and clinical neurophysiology, 79(3), 204-210.

    """
    x = np.array(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def sample_entropy(x, order=2, metric='chebyshev'):
    """Sample Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with KDTree. The list of available
        metric functions is given by: `KDTree.valid_metrics`.

    Returns
    -------
    se : float
        Sample Entropy.

    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.

    Sample entropy of a signal :math:`x` is defined as:

    .. math:: H(x, m, r) = -log\\frac{C(m + 1, r)}{C(m, r)}

    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a Chebyshev distance inferior to :math:`r` and
    :math:`C(m, r)` is the number of embedded vectors of length
    :math:`m` having a Chebyshev distance inferior to :math:`r`.

    Note that if metric == 'chebyshev' and x.size < 5000 points, then the
    sample entropy is computed using a fast custom Numba script. For other
    metric types or longer time-series, the sample entropy is computed using
    a code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort (requires sklearn).

    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    """
    x = np.asarray(x, dtype=np.float64)

    phi = _app_samp_entropy(x, order=order, metric=metric,
                            approximate=False)
    return -np.log(np.divide(phi[1], phi[0]))

