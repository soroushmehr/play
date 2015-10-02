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
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import (TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from cle.cle.utils import segment_axis

from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift,
                        FilterSources)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from scikits.samplerate import resample

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                     GMMMLP)
from play.extensions.sample import Speak

from play.datasets.blizzard import Blizzard

###################
# Define parameters of the model
###################

batch_size = 64
frame_size = 128
k = 20
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 650

depth_theta = 4
hidden_size_mlp_theta = 650
hidden_size_recurrent = 3000

lr = 0.0001

config.recursion_limit = 100000
floatX = theano.config.floatX

save_dir = "/data/lisatmp3/sotelo/results/blizzard/"
#save_dir = "/Tmp/sotelo/results/nips15/fruit"
experiment_name = "blizzard_l0_0"

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
    x = tuple([numpy.array([segment_axis(x, frame_size, 0) for x in var]) for var in data])
    return x

def _downsample_and_upsample(data):
    # HARDCODED
    data_aug = numpy.hstack([data[0], numpy.zeros((batch_size,4))])
    ds = numpy.array([resample(x, 0.5, 'sinc_best') for x in data_aug])
    us = numpy.array([resample(x, 2, 'sinc_best')[:-1] for x in ds])
    return (us,)

def _equalize_size(data):
    min_size = [min([len(x) for x in sequences]) for sequences in zip(*data)]
    x = tuple([numpy.array([x[:size] for x,size in zip(var,min_size)]) for var in data])
    return x

def _get_residual(data):
    # The order is correct?
    ds = numpy.array([x[0]-x[1] for x in zip(*data)])
    return (ds,)

dataset = Blizzard(which_sets = ('train',))

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))

#x_tr = next(data_stream.get_epoch_iterator())

data_stats = numpy.load('/data/lisatmp3/sotelo/data/blizzard/blizzard_standardize.npz')
data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

data_stream = ScaleAndShift(data_stream, scale = 1/data_std, 
                                        shift = -data_mean/data_std)

# downsample and upsample

data_stream = Mapping(data_stream, _downsample_and_upsample, add_sources=('upsampled',))
data_stream = Mapping(data_stream, _equalize_size)
data_stream = Mapping(data_stream, _get_residual, add_sources = ('residual',))
data_stream = FilterSources(data_stream, sources = ('upsampled', 'residual',))
data_stream = Mapping(data_stream, _segment_axis)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)

#################
# Model
#################

x = tensor.tensor3('upsampled')
y = tensor.tensor3('residual')

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [4*hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

transition = LSTM(
            dim=hidden_size_recurrent)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

mlp_gmm = GMMMLP(mlp = mlp_theta,
                  dim = target_size,
                  k = k,
                  const = 0.00001)

bricks = [mlp_x, transition, mlp_gmm]

for brick in bricks:
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0.)
    brick.initialize()

##############
# Test model
##############

x_g = mlp_x.apply(x)
h = transition.apply(x_g)
mu, sigma, coeff = mlp_gmm.apply(h[-2])

from play.utils import GMM
cost = GMM(y, mu, sigma, coeff)
cost = cost.mean()
cost.name = 'sequence_log_likelihood'

cg = ComputationGraph(cost)
model = Model(cost)

#################
# Algorithm
#################

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches = 10,
    prefix="train")

extensions = extensions=[
    Timing(every_n_batches = 10),
    train_monitor,
    TrackTheBest('train_sequence_log_likelihood', every_n_batches = 10),
    Checkpoint(save_dir+experiment_name+".pkl",
               use_cpickle = True,
               every_n_epochs = 15),
    Checkpoint(save_dir+"best_"+experiment_name+".pkl",
               use_cpickle = True
               ).add_condition(['after_epoch'],
                    predicate=OnLogRecord('train_sequence_log_likelihood_best_so_far')),
    Printing(every_n_batches = 10)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()