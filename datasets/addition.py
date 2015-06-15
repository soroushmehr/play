# Copyright (c) 2012-2013, Razvan Pascanu
# All rights reserved.

from fuel.datasets import Dataset
import numpy
from numpy.random import randint, uniform

import theano

floatX = theano.config.floatX

class AdditionTask(Dataset):
    r"""Dataset for the addition task described in the LSTM paper.
    """
    provides_sources = ('features','targets')
    example_iteration_scheme = None

    def __init__(self, length = 10):
        self.length = length
        super(AdditionTask, self).__init__()

    def get_data(self, state=None, request=None):
        if state is not None or request is not None:
            raise ValueError('No state or request possible for this dataset')
        p0 = randint(int(self.length*.1), size=(1,))
        p1 = randint(int(self.length*.4), size=(1,)) + int(self.length*.1)
        data = uniform(size=(self.length, 2)).astype(floatX)
        data[:, 0] = 0.
        data[p0, 0] = 1.
        data[p1, 0] = 1.

        target = 0.5*(data[p0, 1] + data[p1, 1])

        return (data, target)
