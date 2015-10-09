import sys
import numpy
import os

from multiprocessing import Process

from cle.cle.utils import segment_axis

from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift,
                        FilterSources)
from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.streams import DataStream

from play.datasets.blizzard import Blizzard

from scikits.samplerate import resample

#################
# Prepare dataset
#################

batch_size = 64
frame_size = 128

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _segment_axis(data):
    x = tuple([numpy.array([segment_axis(x, frame_size, 0) for x in var]) for var in data])
    return x

def _downsample_and_upsample(data):
    # HARDCODED
    data_aug = numpy.hstack([data[0], numpy.zeros((batch_size,4))])
    ds = numpy.array([resample(x, 0.5, 'sinc_best') for x in data_aug])
    us = numpy.array([resample(x, 2, 'sinc_best')[:-1] for x in ds])
    return (us,)

def _equalize_size(data):
    min_size = [min([len(x) for x in sequences]) for sequences in zip(*data)]
    x = tuple([numpy.array([x[:size] for x,size in zip(var,min_size)]) for var in data])
    return x

def _get_residual(data):
    # The order is correct?
    ds = numpy.array([x[0]-x[1] for x in zip(*data)])
    return (ds,)

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'blizzard_standardize.npz')

data_stats = numpy.load(data_dir)
data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

def open_stream(which_sets= ('train',), port=5557, num_examples = None):

    dataset = Blizzard(which_sets = which_sets)

    if num_examples == None:
        num_examples = dataset.num_examples

    data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            num_examples, batch_size))

    data_stream = ScaleAndShift(data_stream, scale = 1/data_std, 
                                            shift = -data_mean/data_std)
    data_stream = Mapping(data_stream, _downsample_and_upsample, 
                          add_sources=('upsampled',))
    data_stream = Mapping(data_stream, _equalize_size)
    data_stream = Mapping(data_stream, _get_residual,
                          add_sources = ('residual',))
    data_stream = FilterSources(data_stream, 
                          sources = ('upsampled', 'residual',))
    data_stream = Mapping(data_stream, _segment_axis)
    data_stream = Mapping(data_stream, _transpose)
    data_stream = ForceFloatX(data_stream)

    start_server(data_stream, port=port)

if __name__ == "__main__":
    port = int(sys.argv[1])
    num_valid_examples = 4*64*5
    Process(target=open_stream, args=(('train',), port)).start()
    Process(target=open_stream, args=(('valid',), port + 50, num_valid_examples)).start()