import ipdb
import numpy
import theano
import matplotlib
import os
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
from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from cle.cle.utils import segment_axis


from fuel.transformers import (Mapping, 
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                     GMMMLP)

from play.datasets.blizzard import Blizzard
from play.extensions.sample import Speak
from play.extensions.plot import Plot
from scikits.samplerate import resample

###################
# Define parameters of the model
###################

batch_size = 128 #64 for tpbtt
frame_size = 128
k = 20
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 2000

depth_theta = 4
hidden_size_mlp_theta = 2000
hidden_size_recurrent = 2000

lr = 3e-4

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "deep_m1_2"

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
    x = numpy.array([segment_axis(x, frame_size, 0) for x in data[0]])
    return (x,)

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'blizzard_standardize.npz')

data_stats = numpy.load(data_dir)
data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

dataset = Blizzard(which_sets = ('train',))
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))
data_stream = ScaleAndShift(data_stream, scale = 1/data_std, 
                                        shift = -data_mean/data_std)
data_stream = Mapping(data_stream, _segment_axis)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)
train_stream = data_stream

num_valid_examples = 4*64*5
dataset = Blizzard(which_sets = ('valid',))
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            num_valid_examples, 10*batch_size))
data_stream = ScaleAndShift(data_stream, scale = 1/data_std, 
                                        shift = -data_mean/data_std)
data_stream = Mapping(data_stream, _segment_axis)
data_stream = Mapping(data_stream, _transpose)
data_stream = ForceFloatX(data_stream)
valid_stream = data_stream

#################
# Model
#################

x = tensor.tensor3('features')

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = [LSTM(dim=hidden_size_recurrent, 
                   name = "lstm_{}".format(i) ) for i in range(3)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

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

source_names = [name for name in transition.apply.states if 'states' in name]
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
generator.push_initialization_config()

generator.transition.biases_init = IsotropicGaussian(0.01,1)
generator.transition.push_initialization_config()

generator.initialize()

cost_matrix = generator.cost_matrix(x)
cost = cost_matrix.mean()
cost.name = "sequence_log_likelihood"

cg = ComputationGraph(cost)
model = Model(cost)

#################
# Algorithm
#################

n_batches = 500

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    after_epoch = True,
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     [cost],
     valid_stream,
     after_epoch = True,
     every_n_batches = n_batches,
     #before_first_epoch = False,
     prefix="valid")

extensions=[
    train_monitor,
    valid_monitor,
    TrackTheBest('train_sequence_log_likelihood'),
    Plot(save_dir+experiment_name+".png",
         [['train_sequence_log_likelihood',
           'valid_sequence_log_likelihood']],
         every_n_batches = n_batches,
         email=False),
    Checkpoint(save_dir+experiment_name+".pkl",
               use_cpickle = True,
               every_n_batches = n_batches),
    Checkpoint(save_dir+"best_"+experiment_name+".pkl",
               every_n_batches = n_batches
               ).add_condition(['after_epoch'],
                    predicate=OnLogRecord('train_sequence_log_likelihood_best_so_far')),
    Printing(after_epoch = True, every_n_batches = n_batches,),
    Speak(generator,
           steps=320,
           n_samples = 5,
           mean_data = data_mean,
           std_data = data_std,
           sample_rate = 16000,
           save_name = save_dir + "samples/" + experiment_name)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()