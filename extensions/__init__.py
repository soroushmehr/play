import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

class SaveComputationGraph(SimpleExtension):
    def __init__(self, variable, **kwargs):
        super(SaveComputationGraph, self).__init__(**kwargs)
        variable_graph = ComputationGraph(variable)
        self.theano_function = variable_graph.get_theano_function()

    def do(self, which_callback, *args):
        print "empty"