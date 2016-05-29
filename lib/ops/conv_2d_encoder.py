import lib
import lib.ops.conv2d
import lib.ops.linear

import theano.tensor as T
import lasagne

def Conv2DEncoder(
    name,
    input_n_channels,
    input_size,
    n_pools,
    base_n_filters,
    filter_size,
    output_dim,
    inputs,
    ):

    # Pad input to the nearest power of two
    new_input_size = 1
    while new_input_size < input_size:
        new_input_size *= 2
    if new_input_size > input_size:
        padding = (new_input_size - input_size) / 2
        inputs = lasagne.theano_extensions.padding.pad(
            inputs,
            width=padding,
            batch_ndim=2
        )
        input_size = new_input_size

    n_filters = base_n_filters

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.InputConv',
        input_dim=input_n_channels,
        output_dim=n_filters,
        filter_size=filter_size,
        inputs=inputs,
    ))

    for i in xrange(n_pools):

        # We can use either a conv+maxpool or a strided conv to downsample
        # Conv + max-pool:

        # output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        #     name+'.Conv{}BeforePool'.format(i),
        #     input_dim=n_filters,
        #     output_dim=2*n_filters,
        #     filter_size=3,
        #     inputs=output,
        # ))
        # output = T.signal.pool.pool_2d(output, (2,2), ignore_border=True)

        # Strided conv:

        output = T.nnet.relu(lib.ops.conv2d.Conv2D(
            name+'.Conv{}Strided'.format(i),
            input_dim=n_filters,
            output_dim=2*n_filters,
            filter_size=filter_size,
            inputs=output,
            stride=(2,2)
        ))

        output = T.nnet.relu(lib.ops.conv2d.Conv2D(
            name+'.Conv{}AfterPool'.format(i),
            input_dim=2*n_filters,
            output_dim=2*n_filters,
            filter_size=filter_size,
            inputs=output,
        ))

        n_filters *= 2

    volume = n_filters * (input_size / (2**n_pools))**2
    output = output.reshape((output.shape[0], volume))

    return lib.ops.linear.Linear(
        name+'.Output',
        volume,
        output_dim,
        output,
        initialization='glorot'
    )