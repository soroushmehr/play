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
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from cle.cle.utils import segment_axis

from fuel.datasets.fruit import Fruit
from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift,
                        FilterSources)
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from scikits.samplerate import resample

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                     GMMMLP)
from play.bricks.recurrent import SimpleSequenceAttention
from play.extensions.sample import Speak

###################
# Define parameters of the model
###################

batch_size = 15
frame_size = 100
k = 20
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 650

depth_theta = 4
hidden_size_mlp_theta = 650
hidden_size_recurrent = 3000

depth_context = 4
hidden_size_mlp_context = 650
context_size = 1000

lr = 0.0001

config.recursion_limit = 100000
floatX = theano.config.floatX

save_dir = "/data/lisatmp3/sotelo/results/nips15/fruit/"
#save_dir = "/Tmp/sotelo/results/nips15/fruit"
experiment_name = "fruit_l3_0"

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
    x = tuple([numpy.array([segment_axis(x, frame_size, 0) for x in var]) for var in data])
    return x

def _downsample_and_upsample(data):
    ds = numpy.array([resample(x, 0.5, 'sinc_best') for x in data[0]])
    us = numpy.array([resample(x, 2, 'sinc_best') for x in ds])
    return (us,)

def _equalize_size(data):
    min_size = [min([len(x) for x in sequences]) for sequences in zip(*data)]
    x = tuple([numpy.array([x[:size] for x,size in zip(var,min_size)]) for var in data])
    return x

def _get_residual(data):
    # The order is correct?
    ds = numpy.array([x[0]-x[1] for x in zip(*data)])
    return (ds,)


dataset = Fruit(which_sets = ('train','test'))

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=ShuffledScheme(
            dataset.num_examples, batch_size))

x_tr = next(data_stream.get_epoch_iterator())

# Standardize data
all_data = numpy.array([])
for batch in data_stream.get_epoch_iterator():
    for element in batch[0]:
        all_data = numpy.hstack([all_data, element])
mean_data = all_data.mean()
std_data = all_data.std()

data_stream = ScaleAndShift(data_stream, scale = 1/std_data, 
                                        shift = -mean_data/std_data)
data_stream = Mapping(data_stream, _downsample_and_upsample, add_sources=('upsampled',))
data_stream = Mapping(data_stream, _equalize_size)
data_stream = Mapping(data_stream, _get_residual, add_sources = ('residual',))
data_stream = FilterSources(data_stream, sources = ('upsampled', 'residual',))
data_stream = Mapping(data_stream, _segment_axis)
data_stream = Padding(data_stream)
data_stream = FilterSources(data_stream, sources = ('upsampled','residual', 'residual_mask'))
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

activations_context = [Rectifier()]*depth_context

dims_context = [frame_size] + [hidden_size_mlp_context]*(depth_context-1) + \
         [context_size]

x = tensor.tensor3('residual')
x_mask = tensor.matrix('residual_mask')
context = tensor.tensor3('upsampled')

mlp_context = MLP(activations = activations_context,
                  dims = dims_context)

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = LSTM(
            dim=hidden_size_recurrent)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

mlp_gmm = GMMMLP(mlp = mlp_theta,
                  dim = target_size,
                  k = k,
                  const = 0.00001)

emitter = GMMEmitter(gmmmlp = mlp_gmm,
                     output_size = frame_size,
                     k = k,
                     name = "emitter")

source_names=['states']
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names,
    emitter=emitter,
    feedback_brick = feedback,
    name="readout")

attention = SimpleSequenceAttention(
              state_names = source_names,
              state_dims = [hidden_size_recurrent],
              attended_dim = context_size)

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              attention = attention,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.initialize()

mlp_context.weights_init = IsotropicGaussian(0.01)
mlp_context.biases_init = Constant(0.)
mlp_context.initialize()

#ipdb.set_trace()
cost_matrix = generator.cost_matrix(x, x_mask,
        attended = mlp_context.apply(context))
cost = cost_matrix.sum()/x_mask.sum()
cost.name = "sequence_log_likelihood"

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
    after_epoch = True,
    prefix="train")

extensions = extensions=[
    train_monitor,
    TrackTheBest('train_sequence_log_likelihood'),
    Printing(after_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()