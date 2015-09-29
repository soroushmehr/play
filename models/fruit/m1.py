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
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                     GMMMLP)
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

lr = 0.001

config.recursion_limit = 100000
floatX = theano.config.floatX

save_dir = "/data/lisatmp3/sotelo/results/nips15/fruit/"
experiment_name = "fruit_m1_0"

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
    x = numpy.array([segment_axis(x, frame_size, 0) for x in data[0]])
    return (x,)

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

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.initialize()

cost_matrix = generator.cost_matrix(x, x_mask)
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
    Speak(generator = generator, 
          every_n_epochs = 15,
          n_samples = 1,
          mean_data = mean_data,
          std_data = std_data),
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