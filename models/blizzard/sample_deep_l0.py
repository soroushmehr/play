import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import numpy
from blocks.serialization import load
from cle.cle.utils import segment_axis
from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift,
                        FilterSources)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from play.datasets.blizzard import Blizzard
from scikits.samplerate import resample
from scipy.io import wavfile

batch_size = 64
frame_size = 128
n_iter = 4
n_samples = 5

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

data_stats = numpy.load('/data/lisatmp3/sotelo/data/blizzard/blizzard_standardize.npz')
#data_stats = numpy.load('/scratch/jvb-000-aa/sotelo/data/blizzard/blizzard_standardize.npz')

data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

which_sets= ('test',)

dataset = Blizzard(which_sets = which_sets)

data_stream = DataStream.default_stream(
        dataset, iteration_scheme=SequentialScheme(
        dataset.num_examples, batch_size))

epoch_iterator = data_stream.get_epoch_iterator()
raw_audio = next(epoch_iterator)[0]

for i in xrange(n_iter-1):
    x_tr = next(epoch_iterator)[0]
    raw_audio = numpy.hstack([raw_audio, x_tr])

save_dir = "/data/lisatmp3/sotelo/results/blizzard/samples/"

rate = 16000

for i, sample in enumerate(raw_audio[:n_samples]):
    pyplot.plot(sample)
    pyplot.savefig(save_dir +"original_%i.png" % i)
    pyplot.close()

    wavfile.write(save_dir + "original_{}.wav".format(i),
        rate, sample)

data_stream = ScaleAndShift(data_stream, scale = 1/data_std, 
                                        shift = -data_mean/data_std)
data_stream = Mapping(data_stream, _downsample_and_upsample, 
                      add_sources=('upsampled',))

epoch_iterator = data_stream.get_epoch_iterator()

raw_audio_std, upsampled_audio = next(epoch_iterator)

for i in xrange(n_iter-1):
    x_tr,y_tr = next(epoch_iterator)
    raw_audio_std = numpy.hstack([raw_audio_std, x_tr])
    upsampled_audio = numpy.hstack([upsampled_audio, y_tr])

for i,(original_, upsampled_) in enumerate(
                                zip(raw_audio_std, upsampled_audio)[:n_samples]):

    f, (ax1, ax2) = pyplot.subplots(2, sharex=True, sharey=True)
    ax1.plot(original_)
    ax2.plot(upsampled_)
    f.subplots_adjust(hspace=0)
    f.savefig(save_dir + "comparison_upsample_%i.png" % i)
    pyplot.close()

real_residual = raw_audio_std - upsampled_audio

rate = 16000

upsampled_audio_std = upsampled_audio*data_std + data_mean

for i, sample in enumerate(upsampled_audio_std[:n_samples]):
    wavfile.write(save_dir + "upsampled_{}.wav".format(i),
        rate, sample.astype('int16'))

upsampled = _segment_axis((upsampled_audio,))[0]
upsampled = _transpose((upsampled,))[0]

l0_loop = load('/data/lisatmp3/sotelo/results/blizzard/pkl/deep_l0_5557_0.000251530497588.pkl')
predict = l0_loop.extensions[-1].theano_function

residuals = predict(upsampled)[0]
residuals = _transpose((residuals,))[0]
residuals = numpy.array([x.flatten() for x in residuals])

for i,(real_x, predict_x) in enumerate(zip(real_residual, residuals)[:n_samples]):
    
    f, (ax1, ax2) = pyplot.subplots(2, sharex=True, sharey=True)
    ax1.plot(real_x)
    ax2.plot(predict_x)
    f.subplots_adjust(hspace=0)
    f.savefig(save_dir + "residuals_%i.png" % i)
    pyplot.close()

    audio = real_x * data_std

    wavfile.write(save_dir + "real_residual_{}.wav".format(i),
        rate, audio.astype('int16'))

    audio = predict_x * data_std

    wavfile.write(save_dir + "predicted_residual_{}.wav".format(i),
        rate, audio.astype('int16'))

residuals = predict(upsampled)[0]
reconstructed = upsampled + residuals
reconstructed = _transpose((reconstructed,))[0]
reconstructed = numpy.array([x.flatten() for x in reconstructed])
reconstructed_std = reconstructed*data_std + data_mean

for i, sample in enumerate(reconstructed_std[:n_samples]):
    pyplot.plot(sample)
    pyplot.savefig(save_dir + "reconstructed_with_l0_%i.png" % i)
    pyplot.close()

    wavfile.write(save_dir + "reconstructed_with_l0_{}.wav".format(i),
        rate, sample.astype('int16'))




