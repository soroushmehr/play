import ipdb
import numpy
import theano
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Scale,
                               RMSProp, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, Initializable, MLP,
                        Rectifier, Activation, Identity,
                        Random)
from blocks.bricks.base import application
from blocks.bricks.sequence_generators import (AbstractEmitter, 
                        Readout, SequenceGenerator,
                        AbstractFeedback)
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.extensions import FinishAfter, Printing, SimpleExtension
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                    DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from cle.cle.cost import Gaussian
from cle.cle.utils import segment_axis

from fuel.datasets.fruit import Fruit
from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from theano import tensor, config, function

from extensions.sample import Speak

###################
# Define parameters of the model
###################

batch_size = 15
frame_size = 100
target_size = frame_size

depth_x = 4
hidden_size_mlp_x = 650

depth_theta = 4
hidden_size_mlp_theta = 650
hidden_size_recurrent = 3000

lr = 0.001

config.recursion_limit = 100000
floatX = theano.config.floatX

save_dir = "/data/lisatmp3/sotelo/results/nips15/fruit/"
experiment_name = "fruit_m0_1"

#################
# Utils
#################

class SoftPlus(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)

class DeepTransitionFeedback(AbstractFeedback, Initializable):
    def __init__(self, mlp, **kwargs):
        super(DeepTransitionFeedback, self).__init__(**kwargs)

        self.mlp = mlp
        self.feedback_dim = mlp.output_dim
        self.children = [self.mlp]

    @application
    def feedback(self, outputs):
        return self.mlp.apply(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(DeepTransitionFeedback, self).get_dim(name)

class GaussianMLP(Initializable):
    """An mlp brick that branchs out to output
    sigmoid and mu for Gaussian dist
    Parameters
    ----------
    mlp: MLP brick
        the main mlp to wrap around.
    dim:
        output dim
    """

    def __init__(self, mlp, dim, const=0., **kwargs):
        super(GaussianMLP, self).__init__(**kwargs)

        self.dim = dim
        self.const = const
        input_dim = mlp.output_dim
        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, dim],
                      weights_init=self.weights_init,
                      biases_init=self.biases_init,
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, dim],
                         weights_init=self.weights_init,
                         biases_init=self.biases_init,
                         name=self.name + "_sigma")

        self.mlp = mlp
        self.children = [self.mlp, self.mu, self.sigma]
        self.children.extend(self.mlp.children)

    @application
    def apply(self, inputs):
        state = self.mlp.apply(inputs)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const

        return mu, sigma

    @property
    def output_dim(self):
        return self.dim

class GaussianEmitter(AbstractEmitter, Initializable, Random):
    """A Gaussian emitter for the case of real outputs.
    Parameters
    ----------
    initial_output :
        The initial output.
    """
    def __init__(self, gaussianmlp, **kwargs):
        super(GaussianEmitter, self).__init__(**kwargs)
        self.gaussianmlp = gaussianmlp
        self.children = [self.gaussianmlp]

    def components(self, readouts):
        # Returns Mu and Sigma
        return self.gaussianmlp.apply(readouts)

    @application
    def emit(self, readouts):
        mu, sigma = self.components(readouts)
        nr = self.theano_rng.normal(size=mu.shape,
                    avg=mu, std=sigma, dtype=floatX)
        return nr

    @application
    def cost(self, readouts, outputs):
        mu, sigma = self.components(readouts)
        return Gaussian(outputs, mu, sigma)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, frame_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gaussianmlp.output_dim
        return super(GaussianEmitter, self).get_dim(name)

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
	x = numpy.array([segment_axis(x, frame_size, 0) for x in data[0]])
	return (x,)

#dataset = Fruit(which_sets = ('train','test'))
dataset = Fruit(which_sets = ('apple',))

data_stream = DataStream.default_stream(
        	dataset, iteration_scheme=ShuffledScheme(
            dataset.num_examples, batch_size))

x_tr = next(data_stream.get_epoch_iterator())
#ipdb.set_trace()

# Standardize data
all_data = numpy.array([])
for batch in data_stream.get_epoch_iterator():
    for element in batch[0]:
        all_data = numpy.hstack([all_data, element])
mean_data = all_data.mean()
std_data = all_data.std()

data_stream = ScaleAndShift(data_stream, scale = 1/std_data, 
                                        shift = -mean_data/std_data)
data_stream = Mapping(data_stream, _segment_axis)
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)

#################
# Model
#################

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

x = tensor.tensor3('features')
x_mask = tensor.matrix('features_mask')

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = LSTM(
            dim=hidden_size_recurrent)

mlp_theta = MLP( activations = activations_theta,
			 dims = dims_theta)

mlp_gaussian = GaussianMLP(mlp = mlp_theta,
                            dim = target_size,
                            const = 0.00001)

emitter = GaussianEmitter(gaussianmlp = mlp_gaussian,
                          name = "emitter")

source_names=['states']
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names,
    emitter=emitter,
    feedback_brick = feedback,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)

generator.initialize()
cost_matrix = generator.cost_matrix(x, x_mask)
cost = cost_matrix.sum()/x_mask.sum()
cost.name = "sequence_log_likelihood"

##############
# Test with first batch
##############

x_tr, x_mask_tr = next(data_stream.get_epoch_iterator())
f1 = function([x, x_mask], cost)
#print f1(x_tr, x_mask_tr)

#ipdb.set_trace()

################
# Optimization Algorithm
################

cg = ComputationGraph(cost)
model = Model(cost)

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]),
    on_unused_sources='warn')

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    after_epoch = True,
    prefix="train")

extensions = extensions=[
    train_monitor,
    TrackTheBest('train_sequence_log_likelihood'),
    Speak(generator = generator, 
          every_n_epochs = 15,
          n_samples = 1),
    Checkpoint(save_dir+experiment_name+".pkl",
               use_cpickle = True,
               every_n_epochs = 15),
    Checkpoint(save_dir+"best_"+experiment_name+".pkl",
               use_cpickle = True
               ).add_condition(['after_epoch'],
                    predicate=OnLogRecord('train_sequence_log_likelihood_best_so_far')),
    Printing(after_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()