import matplotlib.pyplot as plt
from collections import namedtuple
import copy, numpy as np

RNNArgs = namedtuple('RNNArgs', ['alpha', 'input_dim', 'hidden_dim','output_dim'])

# compute sigmoid non-linearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# Easy derivation compute
def sigmoid_to_derivative(output):
    return output * (1-output)


def main(num_epochs, print_epochs, plot_sample, rnn_args):
    # This dictionary lets us map from int to binary
    int2binary = {}

    # This example will consider 32-bit numbers
    binary_dim = 8
    largest_number = pow(2, binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

    for i in range(largest_number):
        int2binary[i] = binary[i]

    # RNN
    alpha = rnn_args.alpha
    in_dim = rnn_args.input_dim
    hid_dim = rnn_args.hidden_dim
    out_dim = rnn_args.output_dim

    #initialize nn weights
    syn_0 = 2*np.random.random((in_dim, hid_dim)) - 1
    syn_1 = 2*np.random.random((hid_dim, out_dim)) - 1
    syn_h = 2*np.random.random((hid_dim, hid_dim)) - 1

    # and update values
    syn_0_update = np.zeros_like(syn_0)
    syn_1_update = np.zeros_like(syn_1)
    syn_h_update = np.zeros_like(syn_h)

    error_list =[]

    # RNN Training for teaching how to sum 2 binary values
    for epoch in range(num_epochs):
        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number/2)
        a = int2binary[a_int]

        b_int = np.random.randint(largest_number/2)
        b = int2binary[b_int]

        # Expected answer
        c_int = a_int + b_int
        c = int2binary[c_int]

        # d stores the network output (binary encoded)
        tmp_output = np.zeros_like(c)

        overall_error = 0

        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(hid_dim))

        # Feed-forward
        for position in range(binary_dim):

            #generate input and output
            X = np.array([[a[binary_dim-position-1], b[binary_dim-position-1]]])
            y = np.array([[c[binary_dim-position-1]]]).T

            # hidden layer (input ~+ prev_hidden)
            # RNN happens here!
            layer_1 = sigmoid(np.dot(X, syn_0) + np.dot(layer_1_values[-1], syn_h))

            # output layer (new binary representation)
            layer_2 = sigmoid(np.dot(layer_1, syn_1))

            # Calculate the error, SGD
            layer_2_error = y - layer_2
            layer_2_deltas.append((layer_2_error) * sigmoid_to_derivative(layer_2))
            overall_error += np.abs(layer_2_error[0])

            # decode estimate so we can print it out
            tmp_output[binary_dim - position - 1] = np.round(layer_2[0][0])

            # store hidden layer so we can use it in the next timestep
            layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(hid_dim)

        # Back-propagation
        for position in range(binary_dim):
            X = np.array([[a[position], b[position]]])
            layer_1 = layer_1_values[-position-1]
            prev_layer_1 = layer_1_values[-position-2]

            # Error at output layer
            layer_2_delta = layer_2_deltas[-position-1]
            # Error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(syn_h.T)+layer_2_delta.dot(syn_1.T)) * sigmoid_to_derivative(layer_1)

            #let's update all the weights for previous times
            syn_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            syn_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            syn_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        # Update the values according to the learning rate
        syn_0 += syn_0_update * alpha
        syn_1 += syn_1_update * alpha
        syn_h += syn_h_update * alpha

        # Reset the updates for the next epoch
        syn_0_update *= 0
        syn_1_update *= 0
        syn_h_update *= 0

        error_list.append(overall_error[0])

        #print progress every 1000th epoch
        if(epoch%print_epochs == 0):
            print 'Error:   {}'.format(overall_error[0])
            print 'Pred:    {}'.format(tmp_output)
            print 'True:    {}'.format(c)
            out = 0
            for index,x in enumerate(reversed(tmp_output)):
                out += x*pow(2,index)
            print '{} + {} = {}'.format(str(a_int), str(b_int), str(out))

    print_error = error_list[0::plot_sample]
    plt.plot(error_list, color='orange')
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

if __name__=='__main__':
    num_epochs = 30000
    print_epochs = 1000
    plot_sample = 1000
    rnn = RNNArgs(alpha=0.1, input_dim=2, hidden_dim=16, output_dim=1)
    main(num_epochs, print_epochs, plot_sample, rnn)