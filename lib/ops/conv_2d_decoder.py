import lib
import lib.ops.conv2d
import lib.ops.linear
import lib.ops.deconv2d

import theano.tensor as T

def _perforated_upsample(inputs):
    output_size = (2*inputs.shape[2], 2*inputs.shape[3])
    output = T.zeros(
        (inputs.shape[0], inputs.shape[1], output_size[0], output_size[1]),
        dtype='float32'
    )
    output = T.inc_subtensor(output[:,:,::2,::2], inputs)
    return output

def Conv2DDecoder(
    name,
    input_dim,
    n_unpools,
    base_n_filters,
    filter_size,
    output_size,
    output_n_channels,
    inputs
):

    # Pad output size to the nearest power of two
    new_output_size = 1
    while new_output_size < output_size:
        new_output_size *= 2
    if new_output_size > output_size:
        padding = (new_output_size - output_size) / 2
        output_size = new_output_size
    else:
        padding = None

    n_filters = base_n_filters * (2**n_unpools)

    volume = n_filters * (output_size/(2**n_unpools))**2
    output = T.nnet.relu(lib.ops.linear.Linear(
        name+'.Input',
        input_dim=input_dim,
        output_dim=volume,
        inputs=inputs,
        initialization='glorot_he'
    ))
    output = output.reshape((
        output.shape[0],
        n_filters,
        output_size/(2**n_unpools), 
        output_size/(2**n_unpools)
    ))


    for i in xrange(n_unpools):
        output = T.nnet.relu(lib.ops.conv2d.Conv2D(
            name+'.Conv{}BeforeUnpool'.format(i),
            input_dim=n_filters,
            output_dim=n_filters,
            filter_size=filter_size,
            inputs=output,
        ))

        # We can either do a perforated upsample and then a convolution,
        # or a fractionally-strided convolution ('deconv').
        # Perforated upsample + conv:

        # output = _perforated_upsample(output)
        # output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        #     name+'.Conv{}AfterUnpool'.format(i),
        #     input_dim=n_filters,
        #     output_dim=n_filters/2,
        #     filter_size=3,
        #     inputs=output,
        # ))

        # Fractionally-strided conv:

        output = T.nnet.relu(lib.ops.deconv2d.Deconv2D(
            name+'.Deconv{}'.format(i),
            input_dim=n_filters,
            output_dim=n_filters/2,
            filter_size=filter_size,
            inputs=output
        ))

        n_filters /= 2

    output = lib.ops.conv2d.Conv2D(
        name+'.OutputConv',
        input_dim=n_filters,
        output_dim=output_n_channels,
        filter_size=filter_size,
        inputs=output,
        he_init=False
    )

    if padding is not None:
        return output[:,:,padding:-padding, padding:-padding]
    else:
        return output