from theano import tensor
from theano.ifelse import ifelse
from blocks.utils import shared_like
from blocks.monitoring.aggregation import AggregationScheme, Aggregator

class MeanAndVariance(AggregationScheme):
    """Aggregation scheme which computes the mean and variance.
    Parameters
    ----------
    numerator : :class:`~tensor.TensorVariable`
        Theano variable for the numerator e.g. the likelihood
    denominator : :class:`~tensor.TensorVariable`
        Theano variable for the denominator e.g. the batch size
    """
    def __init__(self, numerator, denominator, axis = ()):
        self.axis = ()
        self.numerator = numerator.sum(axis = axis)
        self.denominator = denominator
        self.squared_num = (numerator**2).sum(axis = axis)

    def get_aggregator(self):
        initialized = shared_like(0.)
        numerator_acc = shared_like(self.numerator)
        denominator_acc = shared_like(self.denominator)
        squared_num_acc = shared_like(self.squared_num)

        conditional_update_num = ifelse(initialized,
                                        self.numerator + numerator_acc,
                                        self.numerator)
        conditional_update_den = ifelse(initialized,
                                        self.denominator + denominator_acc,
                                        self.denominator)
        conditional_update_sqn = ifelse(initialized,
                                        self.squared_num + squared_num_acc,
                                        self.squared_num)

        initialization_updates = [(numerator_acc,
                                   tensor.zeros_like(numerator_acc)),
                                  (denominator_acc,
                                   tensor.zeros_like(denominator_acc)),
                                  (squared_num_acc,
                                   tensor.zeros_like(squared_num_acc)),
                                  (initialized, 0.)]
        accumulation_updates = [(numerator_acc,
                                 conditional_update_num),
                                (denominator_acc,
                                 conditional_update_den),
                                (squared_num_acc,
                                 conditional_update_sqn),
                                (initialized, 1.)]
        readout_variable = tensor.stacklists([(numerator_acc /
                                                denominator_acc),
                                              ((squared_num_acc /
                                                denominator_acc) -
                                               (numerator_acc /
                                                denominator_acc)**2)])
        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable = readout_variable)
        return aggregator
