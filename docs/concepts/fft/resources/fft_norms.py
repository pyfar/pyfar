# %%
import pyfar as pf
import matplotlib.pyplot as plt
import numpy as np

pf.plot.use()
# %%
n_samples = 1e3
sampling_rate = 1e4
order = 3
impulse = pf.signals.impulse(
    n_samples, sampling_rate=sampling_rate)
filter = pf.dsp.filter.fractional_octave_bands(
    impulse, num_fractions=1, freq_range=(500, 700))
ir = pf.dsp.filter.fractional_octave_bands(
    impulse, num_fractions=1, freq_range=(200, 400), order=3)*n_samples/2
sine = pf.signals.sine(1e3, n_samples, sampling_rate=sampling_rate)
noise = pf.signals.noise(
    n_samples, rms=1/np.sqrt(2), sampling_rate=sampling_rate)

# %% Non squared norms
plt.close('all')
fft_norms = ['none', 'unitary', 'amplitude', 'rms', 'power', 'psd']
ylim = [(-23, 63), (-23, 63), (-83, 3), (-83, 3), (-83, 3), (-83, 3)]
units = ['\\text{V/Hz}', '\\text{V/Hz}', '\\text{V}', '\\text{V}', '\\text{V}', 'V/\\sqrt{\\text{Hz}}']
log_prefixs = [20, 20, 20, 20, 10, 10]

fig, axes = plt.subplots(int(len(fft_norms)/2), 2, figsize=(30/2.54, 30/2.54), sharex=True)
for idx, (fft_norm, unit, log_prefix, ylim) in enumerate(zip(fft_norms, units, log_prefixs, ylim)):
    ax = plt.subplot(int(len(fft_norms)/2), 2, idx+1)
    filter.fft_norm = fft_norm
    ir.fft_norm = fft_norm
    sine.fft_norm = fft_norm
    noise.fft_norm = fft_norm
    pf.plot.freq(ir, label='IR Signal', log_prefix=log_prefix, ax=ax)
    pf.plot.freq(sine, label='Sine', log_prefix=log_prefix, ax=ax)
    pf.plot.freq(noise, label='Noise', log_prefix=log_prefix, ax=ax)
    pf.plot.freq(filter, label='FIR Filter in dB re $'+unit.replace('V', '1')+'$', log_prefix=log_prefix, ls='--', ax=ax)
    ax.set_ylim(ylim)
    ax.legend(loc='lower left')
    ax.set_ylabel(f'Magnitude in dB re $1 {unit}$')
    if idx % 2 == 1:
        ax.yaxis.set_ticklabels([])
    if idx < 4:
        ax.set_xlabel('')
    ax.set_title('\''+fft_norm+'\'')
fig.tight_layout()
plt.savefig('fft_norms_examples.png', dpi=150)

# %% All norms
plt.close('all')
fft_norms = ['none', 'unitary', 'amplitude', 'rms', 'power', 'psd']
ylim = [(-23, 63), (-23, 63), (-83, 3), (-83, 3), (-153, 3), (-153, 3)]
units = ['V/Hz', 'V/Hz', 'V', 'V', 'V', 'V/sqrt(Hz)']
ylim = [(-23, 63), (-23, 63), (-83, 3), (-83, 3), (-153, 3), (-153, 3)]

fig, axes = plt.subplots(3, 2, figsize=(30/2.54, 30/2.54), sharex=True)
for idx, (fft_norm, unit, ylim) in enumerate(zip(fft_norms, units, ylim)):
    ax = plt.subplot(3, 2, idx+1)
    ir.fft_norm = fft_norm
    sine.fft_norm = fft_norm
    noise.fft_norm = fft_norm
    pf.plot.freq(ir, label='Impulse Response', ax=ax)
    pf.plot.freq(sine, label='Sine', ax=ax)
    pf.plot.freq(noise, label='Noise', ax=ax)
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.set_ylabel('Magnitude in ')
    ax.set_title('\''+fft_norm+'\'')

fig.tight_layout()

# %% standard filters ---------------------------------------------------------
_, ax = plt.subplots(2, 2, figsize=(30/2.54, 20/2.54),
                     sharex=True, sharey=True)
# high-pass
N = 4
frequency = 1e3
btype = 'highpass'
axis = ax[0, 0]
y = pf.dsp.filter.butterworth(impulse, N, frequency, btype=btype)
pf.plot.freq(y, ax=axis, label='Butterworth')

y = pf.dsp.filter.bessel(impulse, N, frequency, btype=btype)
pf.plot.freq(y * 10**(-5/20), ax=axis, label='Bessel')

y = pf.dsp.filter.chebyshev1(impulse, N, 1, frequency, btype=btype)
pf.plot.freq(y * 10**(-10/20), ax=axis, label='Chebyshev Type I')

y = pf.dsp.filter.chebyshev2(impulse, N, 60, 300, btype=btype)
pf.plot.freq(y * 10**(-15/20), ax=axis, label='Chebyshev Type II')

y = pf.dsp.filter.elliptic(impulse, N, 1, 60, frequency, btype=btype)
pf.plot.freq(y * 10**(-20/20), ax=axis, label='Elliptic')

axis.set_title('High-pass filter')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-95, 5)

# low-pass
N = 4
frequency = 1e3
btype = 'lowpass'
axis = ax[0, 1]
y = pf.dsp.filter.butterworth(impulse, N, frequency, btype=btype)
pf.plot.freq(y, ax=axis, label='Butterworth')

y = pf.dsp.filter.bessel(impulse, N, frequency, btype=btype)
pf.plot.freq(y * 10**(-5/20), ax=axis, label='Bessel')

y = pf.dsp.filter.chebyshev1(impulse, N, 1, frequency, btype=btype)
pf.plot.freq(y * 10**(-10/20), ax=axis, label='Chebyshev Type I')

y = pf.dsp.filter.chebyshev2(impulse, N, 60, 3500, btype=btype)
pf.plot.freq(y * 10**(-15/20), ax=axis, label='Chebyshev Type II')

y = pf.dsp.filter.elliptic(impulse, N, 1, 60, frequency, btype=btype)
pf.plot.freq(y * 10**(-20/20), ax=axis, label='Elliptic')

axis.set_title('Low-pass filter')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-95, 5)

# band-pass
N = 4
frequency = [500, 2e3]
btype = 'bandpass'
axis = ax[1, 0]
y = pf.dsp.filter.butterworth(impulse, N, frequency, btype=btype)
pf.plot.freq(y, ax=axis, label='Butterworth')

y = pf.dsp.filter.bessel(impulse, N, frequency, btype=btype)
pf.plot.freq(y * 10**(-5/20), ax=axis, label='Bessel')

y = pf.dsp.filter.chebyshev1(impulse, N, 1, frequency, btype=btype)
pf.plot.freq(y * 10**(-10/20), ax=axis, label='Chebyshev Type I')

y = pf.dsp.filter.chebyshev2(impulse, N, 60, [175, 5500], btype=btype)
pf.plot.freq(y * 10**(-15/20), ax=axis, label='Chebyshev Type II')

y = pf.dsp.filter.elliptic(impulse, N, 1, 60, frequency, btype=btype)
pf.plot.freq(y * 10**(-20/20), ax=axis, label='Elliptic')

axis.set_title('Band-pass filter')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-95, 5)

# band-stop
N = 4
frequency = [250, 5e3]
btype = 'bandstop'
axis = ax[1, 1]
y = pf.dsp.filter.butterworth(impulse, N, frequency, btype=btype)
pf.plot.freq(y, ax=axis, label='Butterworth')

y = pf.dsp.filter.bessel(impulse, N, frequency, btype=btype)
pf.plot.freq(y * 10**(-5/20), ax=axis, label='Bessel')

y = pf.dsp.filter.chebyshev1(impulse, N, 1, frequency, btype=btype)
pf.plot.freq(y * 10**(-10/20), ax=axis, label='Chebyshev Type I')

y = pf.dsp.filter.chebyshev2(impulse, N, 60, [650, 2e3], btype=btype)
pf.plot.freq(y * 10**(-15/20), ax=axis, label='Chebyshev Type II')

y = pf.dsp.filter.elliptic(impulse, N, 1, 60, frequency, btype=btype)
pf.plot.freq(y * 10**(-20/20), ax=axis, label='Elliptic')

axis.set_title('Band-stop filter')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-95, 5)
axis.legend(loc=3)

plt.tight_layout()
plt.savefig('filter_types_standard.png', dpi=150)

# %% filter banks -------------------------------------------------------------

_, ax = plt.subplots(1, 2, figsize=(30/2.54, 12/2.54), sharey=True)

# DIN Filterbank
axis = ax[0]
y = pf.dsp.filter.fractional_octave_bands(impulse, 1, freq_range=(60, 12e3))
pf.plot.freq(y, ax=axis)
axis.set_title('Fractional octave bands (room acoustics)')
axis.set_xlim(20, 20e3)
axis.set_ylim(-60, 10)

# Reconstructing Filterbank
axis = ax[1]
y, *_ = pf.dsp.filter.reconstructing_fractional_octave_bands(impulse, 1)
pf.plot.freq(y, ax=axis)
axis.set_title('Fractional octave bands (perfect reconstructing)')
axis.set_xlim(20, 20e3)
axis.set_ylim(-60, 10)

plt.tight_layout()
plt.savefig('filter_types_filterbanks.png', dpi=150)

# %% cross-over ---------------------------------------------------------------
_, ax = plt.subplots(1, 1, figsize=(15/2.54, 12/2.54))

N = 4
frequency = 1e3
axis = ax

y = pf.dsp.filter.crossover(impulse, N, frequency)
pf.plot.freq(y, ax=axis)

axis.set_title('')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-95, 5)

plt.tight_layout()
plt.savefig('filter_types_crossover.png', dpi=150)

# %% audio filter -------------------------------------------------------------
_, ax = plt.subplots(1, 1, figsize=(15/2.54, 12/2.54))

frequency = 1e3
axis = ax
y = pf.dsp.filter.bell(impulse, frequency, 10, 2)
pf.plot.freq(y, ax=axis, label='Bell')

y = pf.dsp.filter.bell(impulse, frequency, -10, 2)
pf.plot.freq(y, ax=axis, label='Bell')

y = pf.dsp.filter.high_shelve(impulse, 4*frequency, 10, 2, 'II')
pf.plot.freq(y * 10**(-20/20), ax=axis, label='High-shelve')

y = pf.dsp.filter.high_shelve(impulse, 4*frequency, -10, 2, 'II')
pf.plot.freq(y * 10**(-20/20), ax=axis, label='High-shelve')

y = pf.dsp.filter.low_shelve(impulse, 1/4*frequency, 10, 2, 'II')
pf.plot.freq(y * 10**(-40/20), ax=axis, label='Low-shelve')

y = pf.dsp.filter.low_shelve(impulse, 1/4*frequency, -10, 2, 'II')
pf.plot.freq(y * 10**(-40/20), ax=axis, label='Low-shelve')

axis.set_title('')
axis.set_xlabel('')
axis.set_xlim(20, 20e3)
axis.set_ylim(-70, 20)
axis.legend(loc=4, ncol=3)

plt.tight_layout()
plt.savefig('filter_types_parametric-eq.png', dpi=150)
