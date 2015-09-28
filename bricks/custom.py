import theano

from theano import tensor, config

from blocks.bricks import (Activation, Initializable, MLP, Random,
                        Identity)
from blocks.bricks.base import application
from blocks.bricks.sequence_generators import (AbstractEmitter, 
                        AbstractFeedback)

from cle.cle.cost import Gaussian

floatX = config.floatX

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
    def __init__(self, gaussianmlp, output_size, **kwargs):
        super(GaussianEmitter, self).__init__(**kwargs)
        self.gaussianmlp = gaussianmlp
        self.children = [self.gaussianmlp]
        self.output_size = output_size

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
        return tensor.zeros((batch_size, self.output_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gaussianmlp.output_dim
        return super(GaussianEmitter, self).get_dim(name)