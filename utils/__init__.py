from theano import tensor as T
import numpy as np
from cle.cle.utils.op import logsumexp
from cle.cle.utils import totuple
from theano.ifelse import ifelse
from cle.cle.utils import predict

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
    
    return nll.reshape(shape_y[:-1], ndim = n_dim-1)

def BivariateGMM(y, mu, sigma, corr, coeff, binary, epsilon = 1e-5):
    """
    Bivariate gaussian mixture model negative log-likelihood
    Parameters
    ----------
    """
    n_dim = y.ndim
    shape_y = y.shape
    y = y.reshape((-1, shape_y[-1]))
    y = y.dimshuffle(0, 1, 'x')

    mu_1 = mu[:,0,:]
    mu_2 = mu[:,1,:]

    sigma_1 = sigma[:,0,:]
    sigma_2 = sigma[:,1,:]

    c_b =  T.sum( T.xlogx.xlogy0(y[:,0,:], binary) +
              T.xlogx.xlogy0(1 - y[:,0,:], 1 - binary), axis = 1)

    inner1 =  (0.5*T.log(1.-corr**2 + epsilon)) + \
                         T.log(sigma_1) + T.log(sigma_2) +\
                         T.log(2. * np.pi)

    Z = (((y[:,1,:] - mu_1)/sigma_1)**2) + (((y[:,2,:] - mu_2) / sigma_2)**2) - \
        (2. * (corr * (y[:,1,:] - mu_1)*(y[:,2,:] - mu_2)) / (sigma_1 * sigma_2))
    inner2 = 0.5 * (1. / (1. - corr**2 + epsilon))
    cost = - (inner1 + (inner2 * Z))

    nll = -logsumexp(T.log(coeff) + cost, axis=1) - c_b
    return nll.reshape(shape_y[:-1], ndim = n_dim-1)