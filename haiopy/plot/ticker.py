from matplotlib.ticker import FixedLocator, FixedFormatter

class FractionalOctaveFormatter(FixedFormatter):
    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = ['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
        elif n_fractions == 3:
            ticks = ['12.5', '16', '20',
                     '25', '31.5', '40',
                     '50', '63', '80',
                     '100', '125', '160',
                     '200', '250', '315',
                     '400', '500', '630',
                     '800', '1k', '1.25k',
                     '1.6k', '2k', '2.5k',
                     '3.15k', '4k', '5k',
                     '6.3k', '8k', '10k',
                     '12.5k', '16k', '20k']
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)


class FractionalOctaveLocator(FixedLocator):
    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = [16, 31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]
        elif n_fractions == 3:
            ticks = [12.5, 16, 20,
                     25, 31.5, 40,
                     50, 63, 80,
                     100, 125, 160,
                     200, 250, 315,
                     400, 500, 630,
                     800, 1e3, 1250,
                     1600, 2e3, 2500,
                     3150, 4e3, 5e3,
                     6300, 8e3, 10e3,
                     12.5e3, 16e3, 20e3]
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)
