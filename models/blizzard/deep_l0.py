import ipdb
import numpy
import theano
import matplotlib
import sys
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.recurrent import LSTM, RecurrentStack

from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                        DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.streams import ServerDataStream

from theano import tensor, config, function

from play.bricks.custom import GMMMLP
from play.datasets.blizzard import Blizzard
from play.extensions import SaveComputationGraph
from play.extensions.plot import Plot
from play.utils import GMM

###################
# Define parameters of the model
###################

batch_size = 64
frame_size = 128
k = 32
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 32*20

depth_lstm = 3

depth_theta = 4
hidden_size_mlp_theta = 32*20
hidden_size_recurrent = 32*20*3

lr = 10 ** (2*numpy.random.rand() - 5)

config.recursion_limit = 100000
floatX = theano.config.floatX

#job_id = 5557
job_id = int(sys.argv[1])

#save_dir = "/data/lisatmp3/sotelo/results/blizzard/"
save_dir = '/scratch/jvb-000-aa/sotelo/results/'

experiment_name = 'deep_l0_{}_{}'.format(job_id, lr)

train_stream = ServerDataStream(('upsampled', 'residual',), 
                  produces_examples = False,
                  port = job_id)

valid_stream = ServerDataStream(('upsampled', 'residual',), 
                  produces_examples = False,
                  port = job_id+50)
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
            dims = dims_x,
            name = "mlp_x")

transition = [LSTM(dim=hidden_size_recurrent, 
                   name = "lstm_{}".format(i) ) for i in range(depth_lstm)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta,
             name = "mlp_theta")

mlp_gmm = GMMMLP(mlp = mlp_theta,
                  dim = target_size,
                  k = k,
                  const = 0.00001,
                  name = "gmm_wrap")

from play.bricks.custom import GMMEmitter
gmm_emitter = GMMEmitter(gmmmlp = mlp_gmm, output_size = frame_size, k = k)

bricks = [mlp_x, transition, gmm_emitter]

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

#cost = GMM(y, mu, sigma, coeff)

cost = gmm_emitter.cost(h[-2], y)
cost = cost.mean()
cost.name = 'sequence_log_likelihood'

emit = gmm_emitter.emit(h[-2])
emit.name = 'emitter'

cg = ComputationGraph(cost)
model = Model(cost)

#################
# Algorithm
#################

n_batches = 139*16

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     [cost],
     valid_stream,
     after_epoch = True,
     #before_first_epoch = False,
     prefix="valid")

extensions = extensions=[
    Timing(every_n_batches = n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_sequence_log_likelihood', after_epoch = True),
    Plot(save_dir+experiment_name+".png",
         [['train_sequence_log_likelihood',
           'valid_sequence_log_likelihood']],
         every_n_batches = 4*n_batches,
         email=False),
    Checkpoint(save_dir+experiment_name+".pkl",
               use_cpickle = True,
               every_n_batches = n_batches*8,
               after_epoch = True),
    Checkpoint(save_dir+"best_"+experiment_name+".pkl",
     after_epoch = True,
     use_cpickle = True
     ).add_condition(['after_epoch'],
          predicate=OnLogRecord('valid_sequence_log_likelihood_best_so_far')),
    Printing(every_n_batches = n_batches, after_epoch = True),
    FinishAfter(after_n_epochs=2),
    SaveComputationGraph(emit)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()