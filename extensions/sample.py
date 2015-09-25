import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.io import wavfile

class Speak(SimpleExtension):
    """Make your model speak
    Parameters
    ----------
    num_frames : int Number of frames to generate
    """
    def __init__(self, generator, steps=320, n_samples = 10, **kwargs):
        super(Speak, self).__init__(**kwargs)
        steps = 300
        sample = ComputationGraph(generator.generate(n_steps=steps, 
            batch_size=n_samples, iterate=True))
        self.sample_fn = sample.get_theano_function()

    def do(self, callback_name, *args):

        sampled_values = self.sample_fn()[-2]
        sampled_values = sampled_values*std_data + mean_data
        sampled_values = sampled_values.swapaxes(0,1)
        sampled_values = numpy.array([ex_.flatten() for ex_ in sampled_values])

        for i, sample in enumerate(sampled_values):
            #Plot example
            pyplot.plot(sample)
            pyplot.savefig("sample_%i.png" % i)
            pyplot.close()

            #Wav file
            wavfile.write("sample_%i.wav" % i, 8000, sample.astype('int16'))
