"""
VAE + Pixel RNN/CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.mnist_256ary
import lib.ops.mlp
import lib.ops.conv_2d_encoder
import lib.ops.conv_2d_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu
import lib.ops.softmax_and_sample

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne

import functools

MODE = '256ary' # binary or 256ary

MODEL = 'pixel_rnn' # either pixel_cnn or pixel_rnn
PIX_DIM = 32
PIXEL_CNN_LAYERS = 4

FC = False
FC_DIM = 1024
FC_LAYERS = 5

CONV_BASE_N_FILTERS = 16
CONV_N_POOLS = 3
CONV_FILTER_SIZE = 3

LATENT_DIM = 128
ALPHA_ITERS = 10000
VANILLA = False
LR = 2e-4

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TIMES = ('iters', 10*500, 1000*500)
# TIMES = ('seconds', 60*30, 60*60*6)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def Encoder(inputs):
    if MODE=='256ary':
        inputs = inputs.astype(theano.config.floatX) * lib.floatX(2./255)
        inputs -= lib.floatX(0.5)

    if FC:
        mu_and_log_sigma = lib.ops.mlp.MLP(
            'Encoder',
            input_dim=N_CHANNELS*HEIGHT*WIDTH,
            hidden_dim=FC_DIM,
            output_dim=2*LATENT_DIM,
            n_layers=FC_LAYERS,
            inputs=inputs.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
        )
        return mu_and_log_sigma[:, ::2], mu_and_log_sigma[:, 1::2]
    else:
        mu_and_log_sigma = lib.ops.conv_2d_encoder.Conv2DEncoder(
            'Encoder',
            input_n_channels=N_CHANNELS,
            input_size=WIDTH,
            n_pools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_dim=2*LATENT_DIM,
            inputs=inputs
        )
        return mu_and_log_sigma[:, ::2], mu_and_log_sigma[:, 1::2]

def Decoder(latents, images):
    # We apply the sigmoid at a later step
    if FC:
        decoder_out = lib.ops.mlp.MLP(
            'Decoder',
            input_dim=LATENT_DIM,
            hidden_dim=FC_DIM,
            output_dim=PIX_DIM*HEIGHT*WIDTH,
            n_layers=FC_LAYERS,
            inputs=latents
        ).reshape((-1, PIX_DIM, HEIGHT, WIDTH))
    else:
        decoder_out = lib.ops.conv_2d_decoder.Conv2DDecoder(
            'Decoder',
            input_dim=LATENT_DIM,
            n_unpools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_size=WIDTH,
            output_n_channels=PIX_DIM,
            inputs=latents
        )

    output = lib.ops.conv2d.Conv2D(
        'InputConv', 
        input_dim=N_CHANNELS, 
        output_dim=PIX_DIM, 
        filter_size=7, 
        inputs=images, 
        mask_type=('a', N_CHANNELS),
        he_init=False
    )

    output = T.concatenate([output, decoder_out], axis=1)

    if MODEL=='pixel_rnn':

        output = lib.ops.diagonal_bilstm.DiagonalBiLSTM(
            'DiagonalBiLSTM', 
            input_dim=2*PIX_DIM, 
            output_dim=PIX_DIM, 
            input_shape=(N_CHANNELS, HEIGHT, WIDTH),
            inputs=output
        )

    elif MODEL=='pixel_cnn':

        for i in xrange(PIXEL_CNN_LAYERS):
            if i==0:
                inp_dim = 2*PIX_DIM
            else:
                inp_dim = PIX_DIM
            output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
                'PixelCNNConv'+str(i),
                input_dim=inp_dim,
                output_dim=PIX_DIM,
                filter_size=3,
                inputs=output,
                mask_type=('b', N_CHANNELS),
            ))

    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
        'OutputConv1', 
        input_dim=PIX_DIM, 
        output_dim=PIX_DIM, 
        filter_size=1, 
        inputs=output, 
        mask_type=('b', N_CHANNELS), 
    ))

    output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
        'OutputConv2', 
        input_dim=PIX_DIM, 
        output_dim=PIX_DIM, 
        filter_size=1, 
        inputs=output, 
        mask_type=('b', N_CHANNELS), 
    ))

    if MODE=='256ary':
        output = lib.ops.conv2d.Conv2D(
            'OutputConv3',
            input_dim=DIM,
            output_dim=256*N_CHANNELS,
            filter_size=1,
            inputs=output,
            mask_type=('b', N_CHANNELS),
            he_init=False
        ).reshape((-1, 256, N_CHANNELS, HEIGHT, WIDTH)).dimshuffle(0,2,3,4,1)
    else:
        output = lib.ops.conv2d.Conv2D(
            'OutputConv3',
            input_dim=DIM,
            output_dim=N_CHANNELS,
            filter_size=1,
            inputs=output,
            mask_type=('b', N_CHANNELS),
            he_init=False
        )

    return output

total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

mu, log_sigma = Encoder(images)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

if MODE=='256ary':
    reconst_cost = T.nnet.categorical_crossentropy(
        T.nnet.softmax(output.reshape((-1,256))),
        inputs.flatten()
    ).mean()
else:
    reconst_cost = T.nnet.binary_crossentropy(
        T.nnet.sigmoid(output), 
        inputs
    ).sum() / inputs.shape[0].astype(theano.config.floatX)


reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(mu, log_sigma)
reg_cost /= lib.floatX(WIDTH*HEIGHT*N_CHANNELS)

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)

sample_fn_latents = T.matrix('sample_fn_latents')
sample_fn_output = Decoder(sample_fn_latents, images)
if MODE=='256ary':
    sample_fn = theano.function(
        [sample_fn_latents, images],
        lib.ops.softmax_and_sample.softmax_and_sample(sample_fn_output)
    )
else:
    sample_fn = theano.function(
        [sample_fn_latents, images],
        T.nnet.sigmoid(sample_fn_output)
    )

def generate_and_save_samples(tag):

    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        image = scipy.misc.toimage(images, cmin=0.0, cmax=1.0)
        image.save('{}_{}.jpg'.format(filename, tag))

    def binarize(images):
        """
        Stochastically binarize values in [0, 1] by treating them as p-values of
        a Bernoulli distribution.
        """
        return (
            np.random.uniform(size=images.shape) < images
        ).astype(theano.config.floatX)

    latents = np.random.normal(size=(100, LATENT_DIM))
    latents = latents.astype(theano.config.floatX)

    if MODE=='256ary':
        dtype = 'int32'
    else:
        dtype = theano.config.floatX
    samples = np.zeros(
        (BATCH_SIZE, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=dtype
    )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                next_sample = sample_fn(latents, samples)
                if MODE=='binary':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    if MODE=='256ary':
        samples = samples / 255.

    save_images(samples, 'samples')

if MODE=='256ary':
    train_data, dev_data, test_data = lib.mnist_256ary.load(
        BATCH_SIZE, 
        BATCH_SIZE
    )
else:
    train_data, dev_data, test_data = lib.mnist_binarized.load(
        BATCH_SIZE, 
        BATCH_SIZE
    )

lib.train_loop.train_loop(
    inputs=[total_iters, images],
    inject_total_iters=True,
    cost=cost,
    prints=[
        ('alpha', alpha), 
        ('reconst', reconst_cost), 
        ('reg', reg_cost)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    test_data=dev_data,
    callback=generate_and_save_samples,
    times=TIMES
)