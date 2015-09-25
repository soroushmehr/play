from blocks.bricks import MLP, Logistic, Softmax, Linear
from blocks.bricks.cost import (
    CategoricalCrossEntropy, MisclassificationRate)
from blocks.initialization import IsotropicGaussian, Constant
from theano import tensor

from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.monitoring.evaluators import DatasetEvaluator
from variance_aggregation import MeanAndVariance
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

import theano
floatX = theano.config.floatX

from numpy import sqrt
from theano.tensor import cast

mnist_train = MNIST(['train'])
mnist_test = MNIST(['test'])

stream_train = DataStream(mnist_train,
    iteration_scheme=SequentialScheme(mnist_train.num_examples, 100))

#normalization = 'bn1'
normalization = 'bn2'
#normalization = 'off'

def normalize(input_, output_dim):

    if normalization == 'off':
        return input_, None, None

    #normed = tensor.clip(normed, -3., 3.)
    #out = G * normed + B
    #params = [G, B]

    if normalization == 'bn1':
    	output = (input_ - input_.mean(
        	axis=0, keepdims=True)) / (input_.std(
            axis=0, keepdims=True) + 1E-6)
    	return output, None, None
    	
    if normalization == 'bn2':

        M = shared_floatx_nans((output_dim,))
        M.name = 'M'
        S = shared_floatx_nans((output_dim,))
        S.name = 'S'

        #M = input_.mean(axis=0, keepdims=True)
        #S = input_.std(axis=0, keepdims=True)
        output = (input_ - M) / (S + 1E-6)
        return output, M, S

x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

l1 = Linear(name='l1', input_dim=784, output_dim=500,
	weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
l2 = Linear(name='l2', input_dim=500, output_dim=500,
	weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
l3 = Linear(name='l3', input_dim=500, output_dim=500,
	weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
l4 = Linear(name='l4', input_dim=500, output_dim=500,
	weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
l5 = Linear(name='l5', input_dim=500, output_dim=10,
	weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))

l1.initialize()
l2.initialize()
l3.initialize()
l4.initialize()
l5.initialize()

a1 = l1.apply(x.flatten(2)/255)
a1.name = 'a1'
n1, M1, S1 = normalize(a1, output_dim = 500)
o1 = Logistic().apply(n1)

a2 = l2.apply(o1)
n2, M2, S2  = normalize(a2, output_dim = 500)
o2 = Logistic().apply(n2)

a3 = l3.apply(o2)
n3, M3, S3  = normalize(a3, output_dim = 500)
o3 = Logistic().apply(n3)

a4 = l4.apply(o3)
n4, M4, S4  = normalize(a4, output_dim = 500)
o4 = Logistic().apply(n4)

a5 = l5.apply(o4)
n5, M5, S5 = normalize(a5, output_dim = 10)
probs = Softmax().apply(n5)

statistics_list=[(M1,S1,a1), (M2,S2,a2), (M3,S3,a3), (M4,S4,a4), (M5,S5,a5)]

# initialize_variables
# for variable (M,S) in variables:
# 	compute M and S in the whole data.

if normalization == 'bn2':
    for m,s,var in statistics_list:
        var.tag.aggregation_scheme = MeanAndVariance(var, var.shape[0], axis = 0)
        init_mn, init_var = DatasetEvaluator([var]).evaluate(stream_train)[var.name]
        m.set_value(init_mn.astype(floatX))
        s.set_value(sqrt(init_var).astype(floatX))

cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
cost.name = 'cost'
error_rate = MisclassificationRate().apply(y.flatten(), probs)
error_rate.name = 'error_rate'

cg = ComputationGraph([cost])
    
parameters = cg.parameters
# add gradient descent to M,S
if normalization == 'bn2':
    for m,s,var in statistics_list:
        parameters.extend([m,s])

algorithm = GradientDescent(
    cost=cost, parameters=parameters, step_rule=Adam(0.01))

#update the M and S with batch statistics
alpha = 0.1
updates = []
if normalization == 'bn2':
    for m,s,var in statistics_list:
        updates.append((m, cast(alpha*m + (1-alpha)*var.mean(axis=0), floatX)))
        updates.append((s, cast(alpha*s + (1-alpha)*var.std(axis=0) , floatX)))

algorithm.add_updates(updates)
# Since this line wont work with the extension to include parameters
# in the gradient descent. Here's an extension that will do the job.

from blocks.extensions import SimpleExtension
from theano import function

class UpdateExtraParams(SimpleExtension):
    """Adjusts shared variable parameter using some function.
    """
    def __init__(self, input1, input2, updates, **kwargs):
        kwargs.setdefault("after_batch", True)
        super(UpdateExtraParams, self).__init__(**kwargs)
        self.updates = updates
        self.func = function([input1, input2],[],
            updates = updates,
            on_unused_input='ignore')

    def do(self, which_callback, *args):
        self.func(args[0]['features'], args[0]['targets'])

extensions=[
        FinishAfter(after_n_epochs=100),
        DataStreamMonitoring(
            [cost, error_rate],
            DataStream(
                mnist_train,
                iteration_scheme=SequentialScheme(
                    mnist_train.num_examples, 500)),
            prefix='train'),
        DataStreamMonitoring(
            [cost, error_rate],
            DataStream(
                mnist_test,
                iteration_scheme=SequentialScheme(
                    mnist_test.num_examples, 500)),
            prefix='test'),
        Printing()]

if normalization == 'bn2':
    extensions = [UpdateExtraParams(x,y,updates)] + extensions

main_loop = MainLoop(
    algorithm,
    stream_train,
    extensions = extensions)
main_loop.run()