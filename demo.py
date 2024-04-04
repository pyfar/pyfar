import pyfar as pf
import numpy as np

signal = pf.signals.noise(128, seed=7)
signal.complex = False
time = pf.dsp.normalize(signal, domain="time")
freq = pf.dsp.normalize(signal, domain="freq")

signal = pf.signals.noise(128, seed=7)
signal.complex = True
time = pf.dsp.normalize(signal, domain="time")
freq = pf.dsp.normalize(signal, domain="freq")

