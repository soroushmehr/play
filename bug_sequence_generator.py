import blocks
import numpy
import theano
import ipdb

INITIAL_OUTPUTS_CONSTANT = 5

dimension = 3
transition_matrix = 2*numpy.eye(dimension)
embed_matrix = 1*numpy.eye(dimension)
initial_states = numpy.zeros((dimension,))

generated_sequence = INITIAL_OUTPUTS_CONSTANT*numpy.ones((dimension,))
generated_sequence.shape = (1,dimension)
n_steps = 5

hidden_states = numpy.ones((1,dimension))

for i in xrange(n_steps):
	input_ = generated_sequence[-1]
    embed_ = numpy.dot(embed_matrix, input_)
	new_state = numpy.dot(transition_matrix, embed_) + hidden_states[-1]
    hidden_states = numpy.vstack([hidden_states, new_state])
	generated_value = new_state
	generated_sequence = numpy.vstack([generated_sequence, generated_value])

print generated_sequence

from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph

from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, TrivialEmitter)

from theano import tensor

class SimpleRecurrent2(SimpleRecurrent):
    @application(outputs=['states'])
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(
            tensor.ones(self.parameters[1][None, :].shape),
            batch_size,
            0)

class TrivialEmitter2(TrivialEmitter):
    @application
    def initial_outputs(self, batch_size):
        return INITIAL_OUTPUTS_CONSTANT*tensor.ones((batch_size, self.readout_dim))

from blocks.bricks.parallel import Fork

transition = SimpleRecurrent2(dim = dimension,
	activation = Identity())

readout = Readout(
    readout_dim=dimension,
    source_names=['states', 'feedback'],
    emitter=TrivialEmitter2(readout_dim = dimension),
    feedback_brick=TrivialFeedback(output_dim = dimension),
    #merge = Merge(),
    post_merge = Identity(),
    merged_dim = dimension,
    name="readout")

generator = SequenceGenerator(
    readout=readout,
    transition=transition,
    fork = Fork(['inputs'], prototype=Identity()),
    weights_init = initialization.Identity(1.),
    biases_init = initialization.Constant(0.),
    name="generator")

generator.push_initialization_config()
generator.transition.transition.weights_init = initialization.Identity(2.)
generator.initialize()

results = generator.generate(n_steps=n_steps, 
            batch_size=1, iterate=True,
            return_initial_states = True)

results_cg = ComputationGraph(results)
results_tf = results_cg.get_theano_function()

generated_sequence_t = results_tf()[1]
generated_sequence_t.shape=(n_steps+1, dimension)
print generated_sequence_t
print generated_sequence



