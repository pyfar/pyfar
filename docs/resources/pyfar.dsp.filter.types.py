# %%
import pyfar as pf
import matplotlib.pyplot as plt

impulse = pf.signals.impulse(44100)

N = 4
frequency = 1e3
btype = 'highpass'

pf.plot.use()

_, ax = plt.subplots(2, 2, figsize=(30/2.54, 20/2.54), sharex=True)

# standard filter
y = pf.dsp.filter.butterworth(impulse, N, frequency, btype=btype)
pf.plot.freq(y, ax=ax[0, 0], label='Butterworth')

y = pf.dsp.filter.bessel(impulse, N, frequency, btype=btype)
pf.plot.freq(y * 10**(-5/20), ax=ax[0, 0], label='Bessel')

y = pf.dsp.filter.chebyshev1(impulse, N, 1, frequency, btype=btype)
pf.plot.freq(y * 10**(-10/20), ax=ax[0, 0], label='Chebyshev Type I')

y = pf.dsp.filter.chebyshev2(impulse, N, 60, 300, btype=btype)
pf.plot.freq(y * 10**(-15/20), ax=ax[0, 0], label='Chebyshev Type II')

y = pf.dsp.filter.elliptic(impulse, N, 1, 60, frequency, btype=btype)
pf.plot.freq(y * 10**(-20/20), ax=ax[0, 0], label='Elliptic')

ax[0, 0].set_title('Standard filter (low-pass examples)')
ax[0, 0].set_xlabel('')
ax[0, 0].set_xlim(20, 20e3)
ax[0, 0].set_ylim(-95, 5)
ax[0, 0].legend(loc=4)

# audio filter
y = pf.dsp.filter.bell(impulse, frequency, 10, 2)
pf.plot.freq(y, ax=ax[0, 1], label='Bell')

y = pf.dsp.filter.bell(impulse, frequency, -10, 2)
pf.plot.freq(y, ax=ax[0, 1], label='Bell')

y = pf.dsp.filter.high_shelve(impulse, 4*frequency, 10, 2, 'II')
pf.plot.freq(y * 10**(-20/20), ax=ax[0, 1], label='High-shelve')

y = pf.dsp.filter.high_shelve(impulse, 4*frequency, -10, 2, 'II')
pf.plot.freq(y * 10**(-20/20), ax=ax[0, 1], label='High-shelve')

y = pf.dsp.filter.low_shelve(impulse, 1/4*frequency, 10, 2, 'II')
pf.plot.freq(y * 10**(-40/20), ax=ax[0, 1], label='Low-shelve')

y = pf.dsp.filter.low_shelve(impulse, 1/4*frequency, -10, 2, 'II')
pf.plot.freq(y * 10**(-40/20), ax=ax[0, 1], label='Low-shelve')

ax[0, 1].set_title('Audio specific filter')
ax[0, 1].set_xlabel('')
ax[0, 1].set_xlim(20, 20e3)
ax[0, 1].set_ylim(-70, 20)
ax[0, 1].legend(loc=4, ncol=3)

# DIN Filterbank
y = pf.dsp.filter.fractional_octave_bands(impulse, 1, freq_range=(20, 16e3))
pf.plot.freq(y, ax=ax[1, 0])
ax[1, 0].set_title('Fractional octave bands (room acoustics)')
ax[1, 0].set_xlim(20, 20e3)
ax[1, 0].set_ylim(-60, 10)

# Reconstructing Filterbank
y, *_ = pf.dsp.filter.reconstructing_fractional_octave_bands(impulse, 1)
pf.plot.freq(y, ax=ax[1, 1])
ax[1, 1].set_title('Fractional octave bands (perfect reconstructing)')
ax[1, 1].set_xlim(20, 20e3)
ax[1, 1].set_ylim(-60, 10)

plt.savefig('pyfar.dsp.filter.types.png', dpi=150)
