"""Custom tick locators and formatters for matplotlib."""
import numpy as np
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    LogLocator,
    MultipleLocator,
    Formatter)


class FractionalOctaveFormatter(FixedFormatter):
    """Formatter for fractional octave bands."""

    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = [
                '16', '31.5', '63', '125', '250', '500',
                '1k', '2k', '4k', '8k', '16k']
        elif n_fractions == 3:
            ticks = [
                '12.5', '16', '20', '25', '31.5', '40',
                '50', '63', '80', '100', '125', '160',
                '200', '250', '315', '400', '500', '630',
                '800', '1k', '1.25k', '1.6k', '2k', '2.5k',
                '3.15k', '4k', '5k', '6.3k', '8k', '10k',
                '12.5k', '16k', '20k']
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)


class FractionalOctaveLocator(FixedLocator):
    """Locator for fractional octave bands."""

    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = [
                16, 31.5, 63, 125, 250, 500,
                1e3, 2e3, 4e3, 8e3, 16e3]
        elif n_fractions == 3:
            ticks = [
                12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                200, 250, 315, 400, 500, 630, 800, 1e3, 1250,
                1600, 2e3, 2500, 3150, 4e3, 5e3, 6300, 8e3, 10e3,
                12.5e3, 16e3, 20e3]
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)


class LogLocatorITAToolbox(LogLocator):
    """Log-locator inspired by the tick labels used in the ITA-Toolbox."""

    def __init__(
        self,
        base=10.0,
        subs=(0.2, 0.4, 0.6, 1),
        numticks=None,
    ):
        super().__init__(
            base=base,
            subs=subs,
            numticks=numticks)


class MultipleFractionLocator(MultipleLocator):
    r"""
    Tick locator for rational fraction multiples of a specified base,
    e.g. :math:`\pi / 2`.

    The locator is used per default for phase plots in pyfar.

    Parameters
    ----------
    nominator : int
        Nominator of the fraction.
    denominator : int
        Denominator of the fraction.
    base : float
        Base value to multiply the fraction with.

    Examples
    --------
    Use the locator to customize a phase plot:

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> signal = pf.signals.impulse(1e3, 10)
        >>> ax = pf.plot.phase(signal)
        >>> # Minor ticks at multiples of pi/8 on y-axis
        >>> ax.yaxis.set_minor_locator(
        ...     pf.plot.ticker.MultipleFractionLocator(
        ...         nominator=1, denominator=8, base=np.pi))

    """

    def __init__(self, nominator=1, denominator=2, base=1):
        super().__init__(base=base * nominator / denominator)
        self._nominator = nominator
        self._denominator = denominator


class MultipleFractionFormatter(Formatter):
    r"""
    Tick formatter for rational fraction multiples of a specified base,
    e.g. :math:`\pi / 2`.

    The formatter is used per default for phase plots in pyfar.

    Parameters
    ----------
    nominator : int
        Nominator of the fraction.
    denominator : int
        Denominator of the fraction.
    base : float
        Base value to multiply the fraction with.
    base_str : str, optional
        String representation of the base to be used in the tick labels.

    Examples
    --------
    Use the formatter to customize a phase plot together with
    :py:func:`MultipleFractionLocator`:

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> signal = pf.signals.impulse(1e3, 10)
        >>> ax = pf.plot.phase(signal)
        >>> # Major ticks at multiples of pi/4 on y-axis
        >>> ax.yaxis.set_major_locator(
        ...     pf.plot.ticker.MultipleFractionLocator(
        ...         nominator=1, denominator=4, base=np.pi))
        >>> ax.yaxis.set_major_formatter(
        ...     pf.plot.ticker.MultipleFractionFormatter(
        ...         nominator=1, denominator=4, base=np.pi, base_str=r'\pi'))

    """

    def __init__(self, nominator=1, denominator=2, base=1, base_str=None):
        super().__init__()
        self._nominator = nominator
        self._denominator = denominator
        self._base = base
        if base_str is not None:
            self._base_str = base_str
        else:
            self._base_str = "{}".format(base)

    def _gcd(self, nom, denom):
        while denom:
            nom, denom = denom, nom % denom
        return nom

    def __call__(self, x, pos=None):  # noqa: ARG002
        """Return the format for tick val *x* at position *pos*."""
        den = self._denominator
        num = int(np.rint(den*x/self._base))
        com = self._gcd(num, den)
        (num, den) = (int(num / com), int(den/com))
        if den == 1:
            if num == 0:
                string = r'$0$'
            elif num == 1:
                string = r'${}$'.format(self._base_str)
            elif num == -1:
                string = r'$-{}$'.format(self._base_str)
            else:
                string = r'${}{}$'.format(num, self._base_str)
        else:
            if num == 1:
                string = r'$\frac{{{}}}{{{}}}$'.format(self._base_str, den)
            elif num == -1:
                string = r'$-\frac{{{}}}{{{}}}$'.format(self._base_str, den)
            else:
                if num > 0:
                    string = r'$\frac{{{}{}}}{{{}}}$'.format(
                        num, self._base_str, den)
                else:
                    string = r'$-\frac{{{}{}}}{{{}}}$'.format(
                        np.abs(num), self._base_str, den)

        return string
