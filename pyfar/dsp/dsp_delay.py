import numpy as np
import pyfar as pf


def fractional_delay_sinc(signal, delay, order=30, side_lobe_rejection=60,
                          mode="cut"):

    # check input -------------------------------------------------------------
    if not isinstance(signal, (pf.Signal)):
        raise ValueError("Input data has to be of type pyfar.Signal")
    if side_lobe_rejection <= 0:
        raise ValueError("The order must be > 0")
    if side_lobe_rejection <= 0:
        raise ValueError("The side lobe rejection must be > 0")

    # separate integer and fractional delay -----------------------------------
    delay_int = np.atleast_1d(delay).astype(int)
    delay_frac = np.atleast_1d(delay - delay_int)
    # force the fractional delay to be positive
    delay_int[delay_frac < 0] -= 1
    delay_frac[delay_frac < 0] += 1

    # get discrete time vector for sinc functions and Kaiser windows ----------
    # Laakso et al. 1996 Eq (21) applied to the fractional part of the delay
    if order % 2:
        M_opt = -(order-1)/2  # int(delay_frac) = 0 in all cases
                              # because 0 <= delay_frac < 1
    else:
        M_opt = np.round(delay_frac) - order / 2

    # discrete time vector
    n = np.zeros(delay_frac.shape + (order + 1,))

    index = np.nditer(delay_frac, ["multi_index"])
    for d_frac, m in zip(index, M_opt):
        n[index.multi_index] = np.arange(m, m + order + 1) - d_frac

    # get the sinc functions (fractional delay filters) -----------------------
    sinc = np.sinc(n)

    return sinc
