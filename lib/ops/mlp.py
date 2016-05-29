import lib
import lib.ops.linear

import theano.tensor as T

def MLP(name, input_dim, hidden_dim, output_dim, n_layers, inputs):
    output = T.nnet.relu(lib.ops.linear.Linear(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        inputs=inputs,
        initialization='he'
    ))

    for i in xrange(1,n_layers-2):
        output = T.nnet.relu(lib.ops.linear.Linear(
            name+'.Hidden'+str(i),
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            inputs=output,
            initialization='he'
        ))

    return lib.ops.linear.Linear(
        name+'.Output', 
        hidden_dim,
        output_dim, 
        output
    )