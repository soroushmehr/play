import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.io import wavfile

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

class Speak(SimpleExtension):
    """Make your model speak
    Parameters
    ----------
    num_frames : int Number of frames to generate
    """
    def __init__(self, generator, steps=320, n_samples = 10, 
            mean_data = 0, std_data = 1, sample_rate = 8000,
            save_name = "sample_", **kwargs):
        super(Speak, self).__init__(**kwargs)
        steps = 300
        sample = ComputationGraph(generator.generate(n_steps=steps, 
            batch_size=n_samples, iterate=True))
        self.sample_fn = sample.get_theano_function()

        self.mean_data = mean_data
        self.std_data = std_data
        self.sample_rate = sample_rate
        self.save_name = save_name

    def do(self, callback_name, *args):

        sampled_values = self.sample_fn()[-2]
        sampled_values = sampled_values*self.std_data + self.mean_data
        sampled_values = sampled_values.swapaxes(0,1)
        sampled_values = numpy.array([ex_.flatten() for ex_ in sampled_values])

        for i, sample in enumerate(sampled_values):
            #Plot example
            pyplot.plot(sample)
            pyplot.savefig(self.save_name +"_%i.png" % i)
            pyplot.close()

            #Wav file
            wavfile.write(self.save_name + "_%i.wav" % i, 
                self.sample_rate, sample.astype('int16'))
