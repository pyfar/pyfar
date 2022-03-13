import numpy as np
from scipy.special import iv as bessel_first_mod
import pyfar as pf


def fractional_delay_sinc(signal, delay, order=30, side_lobe_suppression=60,
                          mode="cut"):

    # check input -------------------------------------------------------------
    if not isinstance(signal, (pf.Signal)):
        raise ValueError("Input data has to be of type pyfar.Signal")
    if side_lobe_suppression <= 0:
        raise ValueError("The order must be > 0")
    if side_lobe_suppression <= 0:
        raise ValueError("The side lobe rejection must be > 0")
    if mode not in ["cut", "cyclic"]:
        raise ValueError(f"mode is '{mode}' but must be 'cut' or 'cyclic'")

    # separate integer and fractional delay -----------------------------------
    delay_int = np.atleast_1d(delay).astype(int)
    delay_frac = np.atleast_1d(delay - delay_int)
    # force delay_frac >= 0 as required by Laakso et al. 1996 Eq. (2)
    mask = delay_frac < 0
    delay_int[mask] -= 1
    delay_frac[mask] += 1

    # compute the sinc functions (fractional delay filters) -------------------
    # Laakso et al. 1996 Eq. (21) applied to the fractional part of the delay
    # NOTE: This is the delay that is added when applying the fractional part
    #       of the delay and has thus to be accounted for when realizing
    #       delay_int
    if order % 2:
        M_opt = delay_frac.astype("int") - (order-1)/2
    else:
        M_opt = np.round(delay_frac) - order / 2

    # get matrix versions of the fractional delay and M_opt
    delay_frac_matrix = np.tile(
        delay_frac[..., np.newaxis],
        tuple(np.ones(delay_frac.ndim, dtype="int")) + (order + 1, ))
    M_opt_matrix = np.tile(
        M_opt[..., np.newaxis],
        tuple(np.ones(M_opt.ndim, dtype="int")) + (order + 1, ))

    # discrete time vector
    n = np.arange(order + 1) + M_opt_matrix - delay_frac_matrix

    sinc = np.sinc(n)

    # get the Kaiser windows --------------------------------------------------
    # (dsp.time_window can not be used because we need to evaluate the window
    #  for non integer values)

    # beta parameter for side lobe rejection according to
    # Oppenheim (2010) Eq. (10.13)
    beta = pf.dsp.kaiser_window_beta(abs(side_lobe_suppression))

    # Kaiser window according to Oppenheim (2010) Eq. (10.12)
    alpha = order / 2
    L = np.arange(order + 1).astype("float") - delay_frac_matrix
    # required to counter operations on M_opt and make sure that the maxima
    # of the underlying continuous sinc function and Kaiser window appear at
    # the same time
    if order % 2:
        L += .5
    else:
        L[delay_frac_matrix > .5] += 1
    Z = beta * np.sqrt(np.array(1 - ((L - alpha) / alpha)**2, dtype="complex"))
    # suppress small imaginary parts
    kaiser = np.real(bessel_first_mod(0, Z)) / bessel_first_mod(0, beta)

    # apply fractional delay --------------------------------------------------
    signal = pf.dsp.convolve(
        signal, pf.Signal(sinc * kaiser, signal.sampling_rate))

    return signal, sinc, kaiser, M_opt
