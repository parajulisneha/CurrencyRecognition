import numpy
import matplotlib.pyplot
import scipy.special


class NeuralNetwork:
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningRate):
        # set number of nodes in each input, hidden, output layer

        self.iNodes = input_nodes

        self.hNodes = hidden_nodes

        self.oNodes = output_nodes

        # link weight matrices, wih and who

        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer

        # w11 w21

        # w12 w22 etc

        self.weight_hiddeninp = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))

        self.weight_hiddenout = numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # learning rate

        self.lr = learningRate

        # activation function is the sigmoid function

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array

        inputs = numpy.array(inputs_list, ndmin=2).T
        print inputs.shape, "========"
        print self.weight_hiddeninp.shape, "===123====="

        targetsvalue = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer

        hidden_inputs = numpy.dot(self.weight_hiddeninp, inputs)

        # calculate the signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer

        final_inputs = numpy.dot(self.weight_hiddenout, hidden_outputs)

        # calculate the signals emerging from final output layer

        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)

        output_errors = targetsvalue - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes

        hidden_errors = numpy.dot(self.weight_hiddenout.T, output_errors)

        # update the weights for the links between the hidden and output layers

        self.weight_hiddenout += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                     numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers

        self.weight_hiddeninp += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                     numpy.transpose(inputs))

        pass

    # query the neural network

    def query(self, inputs_list):
        # convert inputs list to 2d array

        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer

        hidden_inputs = numpy.dot(self.weight_hiddeninp, inputs)

        # calculate the signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer

        final_inputs = numpy.dot(self.weight_hiddenout, hidden_outputs)

        # calculate the signals emerging from final output layer

        final_outputs = self.activation_function(final_inputs)

        return final_outputs

if __name__ == "__main__":
    # number of input, hidden and output nodes

    input_nodes = 2500

    hidden_nodes =  50

    output_nodes = 7

    # learning rate

    learning_rate = 0.1

    # create instance of neural network

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = numpy.genfromtxt('TrainingDataset.csv', delimiter=',', filling_values=0.0)
    new_train_data = training_data_file[:,1:-5]

    print "Train data shape", new_train_data

    epochs = 5

    for e in range(epochs):
        for record in new_train_data:
            inputs = (numpy.asfarray(record) * 0.90) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(record[0])] = 0.99
            nn.train(inputs, targets)
        pass
    pass

    # test the neural network


    # scorecard for how well the network performs, initially empty

    scorecard = []

    # go through all the records in the test data set

    test_data_file = numpy.genfromtxt('TestingDataset.csv', delimiter=',', filling_values=0.0)
    new_test_data = test_data_file[:,1:]
    print "Test data shape", new_test_data.shape

    for record in new_test_data:
        inputs = (numpy.asfarray(record)* 0.99) + 0.01
        outputs = nn.query(inputs)
        label = numpy.argmax(outputs)
        if label == record:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    scorecard_array = numpy.asarray(scorecard)

    print ("performance = ", scorecard_array.sum() / scorecard_array.size)