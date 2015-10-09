import blocks
import numpy
import theano
import ipdb

dimension = 3
transition_matrix = 2*numpy.eye(dimension)
embed_matrix = 1*numpy.eye(dimension)
initial_states = numpy.zeros((dimension,))

generated_sequence = initial_states
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
    SequenceGenerator, Readout)

from theano import tensor

class SimpleRecurrent2(SimpleRecurrent):
    @application(outputs=['states'])
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(
            tensor.ones(self.parameters[1][None, :].shape),
            batch_size,
            0)

from blocks.bricks.parallel import Fork

transition = SimpleRecurrent2(dim = dimension,
	activation = Identity())

readout = Readout(
    readout_dim=dimension,
    source_names=transition.apply.states + ["feedback"],
    name="readout")

generator = SequenceGenerator(
    readout=readout,
    transition=transition,
    fork = Fork(['inputs'], prototype=Identity()),
    weights_init = initialization.Identity(1.),
    biases_init = initialization.Constant(0.),
    name="generator")

generator.push_initialization_config()
#generator.fork.weights_init = initialization.Identity(1.)
generator.transition.transition.weights_init = initialization.Identity(2.)
generator.initialize()

results = generator.generate(n_steps=n_steps, 
            batch_size=2, iterate=True,
            return_initial_states = True)

results_cg = ComputationGraph(results)
results_tf = results_cg.get_theano_function()

generated_sequence_t = results_tf()[1]
generated_sequence_t.shape=(n_steps+1, dimension)
print generated_sequence_t
print generated_sequence

from blocks.bricks.base import application
from blocks.bricks.attention import AbstractAttention

class SimpleSequenceAttention(AbstractAttention):
    """Combines a conditioning sequence and a recurrent transition via
    attention.

    The conditioning sequence should have the shape:
    (seq_length, batch_size, features)

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    """

    @application(outputs=['glimpses', 'step'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                attended_mask=None, step = None, **states):
        return attended[step, tensor.arange(attended.shape[1]), :], step + 1

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended',
                 'attended_mask', 'step'] +
                self.state_names)

    @application(outputs=['glimpses', 'step'])
    def initial_glimpses(self, batch_size, attended = None):
        return ([tensor.zeros((batch_size, self.attended_dim))]
            + [tensor.zeros((batch_size,), dtype='int64')])

    def get_dim(self, name):
        if name == 'step':
            return 0
        if name == 'glimpses':
            return self.attended_dim
        return super(SimpleSequenceAttention, self).get_dim(name)

# seq_length * batch_size * features
batch_size = 2
seq_length = n_steps
features = 3

attended_tr = numpy.array( range(batch_size*seq_length*features)).astype('float32')
attended_tr.shape = (seq_length, batch_size, features)

from theano import tensor, function
from blocks.bricks.attention import AttentionRecurrent

attended = tensor.tensor3('attended')
ssa = SimpleSequenceAttention(['states'],[3],3)

ar = AttentionRecurrent(
    transition = transition,
    attention = ssa,
    )

ar.weights_init = initialization.Constant(0.)
ar.biases_init = initialization.Constant(1.)
ar.initialize()

inputs = tensor.tensor3('inputs')

#ar.apply(attended = attended_tv, n_steps = n_steps, batch_size = 2)
states, glimpses, step = ar.initial_states(1, attended = attended)
glimpses, step =ar.take_glimpses(attended = attended, states = states, glimpses = glimpses, step = step)
states =ar.compute_states(inputs = inputs, attended = attended, states = states, glimpses = glimpses, step = step)
distributed = ar.distribute.apply(inputs = inputs, glimpses = glimpses)
states = ar.compute_states(states = states, inputs = inputs[0], glimpses = glimpses, step = step, attended = attended)

batch_size = 2
features = 3
#input_tr = numpy.zeros((seq_length, batch_size, features)).astype('float32')
input_tr = generated_sequence_t[1:]

try2 = ar.apply(attended = attended, inputs = inputs,
    states = states, glimpses = glimpses, step = step, iterate = False)

try3 = ar.apply(attended = attended, inputs = inputs,
    states = states, glimpses = glimpses, step = step)

function([attended, inputs], try3)(attended_tr, input_tr[:4,:,:])

readout = Readout(
    readout_dim=dimension,
    source_names=transition.apply.states + ["feedback"] + ["glimpses"],
    name="readout")

generator2 = SequenceGenerator(
    readout=readout,
    transition=transition,
    attention = ssa,
    fork = Fork(['inputs'], prototype=Identity()),
    weights_init = initialization.Identity(1.),
    biases_init = initialization.Constant(0.),
    name="generator")

generator2.push_initialization_config()
#generator.fork.weights_init = initialization.Identity(1.)
generator2.transition.transition.weights_init = initialization.Identity(2.)
generator2.initialize()

results = generator2.generate(
            attended = attended,
            n_steps=n_steps,
            batch_size=2, iterate=True,
            return_initial_states = True,)

results_cg = ComputationGraph(results)
results_tf = results_cg.get_theano_function()
results_tf(attended_tr)[0]
results_tf(numpy.zeros(attended_tr.shape, dtype='float32'))[0]

