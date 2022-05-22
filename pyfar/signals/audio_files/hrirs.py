"""
Get selected data from FABIAN HRTF database and copy to pyfar
http://dx.doi.org/10.14279/depositonce-5718.5
"""
import os
import sofar as sf
import pyfar as pf

database = r"C:\Users\panik\Documents\Uni\Measurements\2020_FABIAN_HRTF_DATABASE_v4\1 HRIRs\SOFA"

# read HRIRs
file = os.path.join(database, "FABIAN_HRIR_measured_HATO_0.sofa")
sofa = sf.read_sofa(file)
hrirs, sg, _ = pf.io.convert_sofa(sofa)

# shorten to 128 samples
hrirs_short = pf.dsp.time_shift(hrirs, -10)
hrirs_short = pf.dsp.time_window(hrirs_short, [117, 127], shape="right")
# visual check for most critical position
left, *_ = sg.find_nearest_k(0, 1, 0, )
pf.plot.freq(hrirs[left])
pf.plot.freq(hrirs_short[left], c='k', linestyle="--")

# select horizontal and median plane
_, mask_horizontal = sg.find_slice('elevation', 'deg', 0)
_, mask_median = sg.find_slice('lateral', 'deg', 0)
mask_all = mask_horizontal | mask_median

# remove undesired data from SOFA file
sofa.Data_IR = hrirs_short.time[mask_all]
sofa.SourcePosition = sofa.SourcePosition[mask_all]

# write reduced set of HRIRs
sf.write_sofa('hrirs.sofa', sofa)

# copy inverse CTF: read write to update to latest convention and remove
# optional, attributes (they would produce unnecessary command line output)
file = os.path.join(database, "FABIAN_CTF_measured_inverted_smoothed.sofa")
sofa = sf.read_sofa(file)
sofa.delete("GLOBAL_ListenerShortName")
sofa.delete("GLOBAL_DatabaseName")
sf.write_sofa('hrirs_ctf_inverted_smoothed.sofa', sofa)
