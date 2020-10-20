import numpy as np
import sofa


def generate_test_sofa(dir='tests/test_io_data/'):
    """ Generate the reference sofa files used for testing the read_sofa function.
    Parameters
    -------
    dir : String
        Path to save the reference plots.
    """
    conventions = ['GeneralFIR', 'GeneralTF']
    n_measurements = 1
    n_receivers = 2
    n_samples = 1000

    for convention in conventions:
        sofafile = sofa.Database.create(
                        (dir + convention + '.sofa'),
                        convention, 
                        dimensions={"M": n_measurements, "R": n_receivers, "N": n_samples})
        sofafile.Listener.initialize(fixed=["Position"])
        sofafile.Receiver.initialize(fixed=["Position"])
        sofafile.Source.initialize(variances=["Position"])
        sofafile.Emitter.initialize(fixed=["Position"])

        sofafile.Data.create_attribute("IR",
                        np.random.random_sample(
                            (n_measurements, n_receivers, n_samples))


generate_test_sofa()
