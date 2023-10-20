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

# %% Plot
fft_norms = ['none', 'unitary', 'amplitude', 'rms', 'power', 'psd']
ylim = [(-23, 63), (-23, 63), (-83, 3), (-83, 3), (-83, 3), (-83, 3)]
units = [
    '$\\mathrm{V/Hz}$', '$\\mathrm{V/Hz}$', '$\\mathrm{V}$', '$\\mathrm{V}$',
    '$\\mathrm{V}$', '$V/\\sqrt{\\mathrm{Hz}}$']
log_prefixs = [20, 20, 20, 20, 10, 10]

fig, axes = plt.subplots(3, 2, figsize=(30/2.54, 30/2.54), sharex=True)
for idx, (fft_norm, unit, log_prefix, ylim) in \
        enumerate(zip(fft_norms, units, log_prefixs, ylim)):
    ax = plt.subplot(3, 2, idx+1)
    filter.fft_norm = fft_norm
    impulse.fft_norm = fft_norm
    sine.fft_norm = fft_norm
    noise.fft_norm = fft_norm
    pf.plot.freq(
        impulse, label='Impulse in dB re '+unit.replace('V', '1'),
        log_prefix=log_prefix, ax=ax)
    pf.plot.freq(
        filter, label='FIR Filter in dB re '+unit.replace('V', '1'),
        log_prefix=log_prefix, ax=ax)
    pf.plot.freq(
        noise, label=f'Noise in dB re {unit}', log_prefix=log_prefix, ax=ax)
    pf.plot.freq(
        sine, label=f'Sine in dB re {unit}', log_prefix=log_prefix, ax=ax)

    ax.set_ylim(ylim)
    ax.legend(loc='lower left')
    if idx % 2 == 1:
        ax.yaxis.set_ticklabels([])
        ax.set_ylabel('')
    if idx < 4:
        ax.set_xlabel('')
    ax.set_title('\''+fft_norm+'\'')

plt.savefig('fft_norms_examples.png', dpi=150)
