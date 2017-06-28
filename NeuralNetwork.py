# import inline as inline
import numpy
import matplotlib.pyplot
# %matplotlib inline




# neural network class definition

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


# number of input, hidden and output nodes

input_nodes = 2505

hidden_nodes = 10

output_nodes = 7

# learning rate

learning_rate = 0.1

# create instance of neural network

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list

training_data_file = open("/Users/apple/PycharmProjects/CurrencyRecognitionNN/TrainingDataset.csv", 'r')

training_data_list = training_data_file.readlines()
print len(training_data_list), "----------"

training_data_file.close()

#train the neural network


# epochs is the number of times the training data set is used for training

epochs = 5

for e in range(epochs):

    # go through all records in the training data set

    for record in training_data_list:
        # split the record by the ',' commas

        all_values = record.split(',')
        # new_list = []
        # for item in all_values:
        #     try:
        #         new_list.append(float(item))
        #     except ValueError:
        #         new_list.pop()
        # print new_list, "-------"
        # print all_values, type(all_values)
        # scale and shift the inputs

        inputs = ((numpy.asfarray(all_values[1:]) * 0.90) + 0.01).reshape((50, 50))

        # create the target output values (all 0.01, except the desired label which is 0.99)

        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] is the target label for this record

        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)

        pass

    pass

# load the mnist test data CSV file into a list

test_data_file = open("/Users/apple/PycharmProjects/CurrencyRecognitionNN/TestingDataset.csv", 'r')

test_data_list = test_data_file.readlines()

test_data_file.close()

# test the neural network


# scorecard for how well the network performs, initially empty

scorecard = []

# go through all the records in the test data set

for record in test_data_list:

    # split the record by the ',' commas

    all_values = record.split(',')

    # correct answer is first value

    correct_label = int(all_values[0])

    # scale and shift the inputs

    inputs = (numpy.asfarray(all_values[1:])* 0.99) + 0.01

    # query the network

    outputs = n.query(inputs)

    # the index of the highest value corresponds to the label

    label = numpy.argmax(outputs)

    # append correct or incorrect to list

    if label == correct_label:

        # network's answer matches correct answer, add 1 to scorecard

        scorecard.append(1)

    else:

        # network's answer doesn't match c             orrect answer, add 0 to scorecard

        scorecard.append(0)

        pass

    pass

# calculate the performance score, the fraction of correct answers

scorecard_array = numpy.asarray(scorecard)

print ("performance = ", scorecard_array.sum() / scorecard_array.size)