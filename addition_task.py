import ipdb

from theano import tensor

from blocks.bricks import Linear, Rectifier, Softmax, MLP, Tanh, Identity
from blocks.bricks.cost import SquaredError
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent

from blocks.graph import ComputationGraph

from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

from fuel.transformers import Mapping, Batch
from fuel.schemes import ConstantScheme
from fuel.transformers import Flatten

from extensions.plot import Plot
from datasets.addition import AdditionTask

from numpy import swapaxes

def _transpose(data):
    return tuple(swapaxes(array,0,1) if len(array.shape) > 2 else array for array in data)

dataset = AdditionTask(1000)
train_stream = dataset.get_example_stream()
train_stream = Batch(train_stream, iteration_scheme=ConstantScheme(10))
train_stream = Mapping(train_stream, _transpose)

features_test, targets_test = next(train_stream.get_epoch_iterator())

x = tensor.tensor3('features')
y = tensor.matrix('targets')

n_batchs = 1000
h_dim = 2
x_dim = 2

encode = Linear(name='encode',
                input_dim=x_dim,
                output_dim=h_dim)

gates  = Linear(name = 'gates',
                input_dim = x_dim,
                output_dim = 2*h_dim)

#lstm = LSTM(activation=Tanh(),
#            dim=h_dim, name="lstm")

lstm = SimpleRecurrent(dim=h_dim,
                       activation=Tanh())

#lstm = GatedRecurrent(dim=h_dim,
#                      activation=Tanh())

decode = Linear(name='decode',
                input_dim=h_dim,
                output_dim=1)

for brick in (encode, gates, decode):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0.)
    brick.initialize()

lstm.weights_init = IsotropicGaussian(0.01)
#lstm.weights_init = Orthogonal()
lstm.biases_init = Constant(0.)
lstm.initialize()

#ComputationGraph(encode.apply(x)).get_theano_function()(features_test)[0].shape
#ComputationGraph(lstm.apply(encoded)).get_theano_function()(features_test)
#ComputationGraph(decode.apply(hiddens[-1])).get_theano_function()(features_test)[0].shape

#ComputationGraph(SquaredError().apply(y, y_hat.flatten())).get_theano_function()(features_test, targets_test)[0].shape

encoded = encode.apply(x)
#hiddens = lstm.apply(encoded, gates.apply(x))
hiddens = lstm.apply(encoded)
y_hat  = decode.apply(hiddens[-1])

cost = SquaredError().apply(y, y_hat)
cost.name = 'cost'

#ipdb.set_trace()

#ComputationGraph(y_hat).get_theano_function()(features_test)[0].shape
#ComputationGraph(cost).get_theano_function()(features_test, targets_test)[0].shape

cg = ComputationGraph(cost)

#cg = ComputationGraph(hiddens).get_theano_function()
#ipdb.set_trace()
algorithm = GradientDescent(cost=cost, 
                            params=cg.parameters,
                            step_rule=CompositeRule([StepClipping(5.0),
                                                     Scale(0.01)]))

monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches=100,
    prefix="test")

main_loop = MainLoop(data_stream=train_stream,
					 algorithm=algorithm,
                     extensions=[monitor,
                                 FinishAfter(every_n_batches=20000),
                                 Plot('test2.png', [['test_cost']], every_n_batches = 1000, email = False),
                                 Printing(every_n_batches = 1000)])
main_loop.run()
