import lib

import numpy as np
import theano
import theano.tensor as T
import lasagne

def Conv1D(name, input_dim, output_dim, filter_size, inputs, biases=True):
    """
    inputs.shape: (batch size, input_dim, height)
    output.shape: (batch size, output_dim, height)
    * performs valid convs
    """
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters = lib.param(
        name+'.Filters',
        uniform(
            1./np.sqrt(input_dim * filter_size),
            (output_dim, input_dim, filter_size, 1)
        )
    )

    inputs = inputs.dimshuffle(0, 1, 2, 'x')
    result = T.nnet.conv2d(inputs, filters, border_mode='valid')
    result = T.addbroadcast(result, 3)
    result = result.dimshuffle(0, 1, 2)

    if biases:
        biases_ = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases_[None, :, None]

    return result