import numpy
np = numpy
from fuel.streams import DataStream
from fuel.transformers import Flatten
import theano


def sin_memory_streams(path,
                       seq_len=2,
                       batch_size=200,
                       which_set='train',
                       range_start=0,
                       range_end=None):
    if which_set not in ['train', 'valid', 'test']:
        raise ValueError('Only defined sets are: train, valid, test.')
    from fuel.datasets import IterableDataset
    npzfile = np.load(path)
    data = npzfile[which_set].astype(theano.config.floatX)
    if range_end is None:
        range_end = len(data)
    data = data[range_start:range_end]

    # TODO: batch_size should be a factor of (range_end - range_start)
    #num_batch = (range_end - range_start) / batch_size

    # TODO: seq_len should be a factor of whole sequence length (here: 256)
    #num_timesteps = data.shape[1] / seq_len

    #data:(20000, 256)
    #->(100, 200, 256)
    data = data.reshape((-1, batch_size, data.shape[1]))
    #->(100, 200, 128, 2) [0, 1, 2, 3]
    data = data.reshape((data.shape[0], batch_size, -1, seq_len))
    #->(100, 128, 200, 2) [0, 2, 1, 3]
    data = np.transpose(data, (0, 2, 1, 3))
    #(num_batch, num_timesteps, batch_size, seq_len))
    X_mean = npzfile['train_mean']
    X_std = npzfile['train_std']
    dataset = IterableDataset({'features': data,})
    return DataStream(dataset), X_mean, X_std


if __name__ == '__main__':
    valid, mean, std = sin_memory_streams(path='/Tmp/mehris/sin_lowfreq_2_train-valid-test-mean-std.npz',
                                          seq_len=2,
                                          batch_size=4,
                                          which_set='valid',
                                          range_start=0,
                                          range_end=None)
    for data in valid.get_epoch_iterator():
        print data[0].shape
        break


