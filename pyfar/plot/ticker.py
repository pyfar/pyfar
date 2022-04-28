import numpy as np
from matplotlib import transforms as mtransforms
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    LogFormatter,
    LogLocator,
    MultipleLocator,
    Formatter)


class FractionalOctaveFormatter(FixedFormatter):
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
    def __init__(
        self,
        base=10.0,
        subs=(0.2, 0.4, 0.6, 1),
        numdecs=4,
        numticks=None
    ):
        super().__init__(
            base=base,
            subs=subs,
            numdecs=numdecs,
            numticks=numticks)


class LogFormatterITAToolbox(LogFormatter):
    """
    Log-formatter inspired by the tick labels used in the ITA-Toolbox
    for MATLAB. Uses unit inspired labels e.g. `1e3 = 1k`, `1e6 = 1M`.
    """
    def __init__(
        self,
        base=10.0,
        labelOnlyBase=False,
        minor_thresholds=None,
        linthresh=None
    ):
        super().__init__(
            base=base,
            labelOnlyBase=labelOnlyBase,
            minor_thresholds=minor_thresholds,
            linthresh=linthresh)

    def _num_to_string(self, x, vmin, vmax):
        if x >= 1000 and x < 1e6:
            s = '{:g}k'.format(x/1e3)
        elif x >= 1e6 and x < 1e9:
            s = '{:g}M'.format(x/1e6)
        elif x >= 1e9:
            s = '{:g}G'.format(x/1e9)
        else:
            try:
                s = self._pprint_val(x, vmax - vmin)
            except AttributeError:
                s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val *x*.
        """
        if x == 0.0:  # Symlog
            return '0'

        x = x

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        s = self._num_to_string(x, vmin, vmax)
        return self.fix_minus(s)


class MultipleFractionLocator(MultipleLocator):
    r"""
    Tick locator for rational fraction multiples of a specified base, ie.
    `math: \pi / 2`.
    """
    def __init__(self, nominator=1, denominator=2, base=1):
        super().__init__(base=base * nominator / denominator)
        self._nominator = nominator
        self._denominator = denominator


class MultipleFractionFormatter(Formatter):
    r"""
    Tick formatter for rational fraction multiples of a specified base, ie.
    `math: \pi / 2`.
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

    def __call__(self, x, pos=None):
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
