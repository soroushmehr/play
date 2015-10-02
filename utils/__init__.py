from theano import tensor as T
import numpy as np
from cle.cle.utils.op import logsumexp
from cle.cle.utils import totuple
from theano.ifelse import ifelse

def chunkIt(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

def GMM(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """

    n_dim = y.ndim
    shape_y = y.shape
    y = y.reshape((-1, shape_y[-1]))
    y = y.dimshuffle(0, 1, 'x')

    mu = mu.reshape((-1,mu.shape[-1]/coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((-1, sig.shape[-1]/coeff.shape[-1],coeff.shape[-1]))
    coeff = coeff.reshape((-1, coeff.shape[-1]))

    inner = -0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis= -2)

    nll = -logsumexp(T.log(coeff) + inner, axis=-1)

    # Adjust dimension
    new_dim = T.set_subtensor(shape_y[-1],1)

    nll = nll.reshape(new_dim, ndim = n_dim)
    nll = nll.flatten(n_dim-1)
    
    return nll

