from addition import AdditionTask

from fuel.transformers import Mapping, Batch
from fuel.schemes import ConstantScheme

from numpy import swapaxes

def _transpose(data):
    return tuple(swapaxes(array,0,1) for array in data if len(array.shape) > 2 )

dataset = AdditionTask(17)
data_stream = dataset.get_example_stream()
data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(14))
data_stream = Mapping(data_stream, _transpose)

print next(data_stream.get_epoch_iterator())[0].shape

