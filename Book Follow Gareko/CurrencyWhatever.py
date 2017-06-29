
# coding: utf-8

# In[1]:

import numpy, scipy


# In[2]:

# Global variables:

# number of input, hidden and output nodes
input_nodes = 2500
hidden_nodes =  10
output_nodes = 7

# learning rate
learning_rate = 0.1

# iterations
epochs = 5

# scorecard for output
scorecard = []


# In[3]:

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningRate):
        self.iNodes = input_nodes
        self.hNodes = hidden_nodes
        self.oNodes = output_nodes
        self.lr = learningRate
        
        self.weight_hiddeninp = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.weight_hiddenout = numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
        self.activation_function = lambda x: scipy.special.expit(x)

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


# In[4]:

# Neural Network Instance here
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[5]:

print "Hidden Nodes \t: ", nn.hNodes
print "Input Nodes \t: ", nn.iNodes
print "Output Nodes \t: ", nn.oNodes
print "Learning Rate \t: ", nn.lr
print "weight_hiddeninp shape: ", nn.weight_hiddeninp[0].shape
print "weight_hiddenout shape: ", nn.weight_hiddenout[0].shape


# In[6]:

# training data
training_data_file = numpy.genfromtxt('TrainingDataset.csv', delimiter=',', filling_values=0.0)
new_train_data = training_data_file[:,1:-5]

# testing data
test_data_file = numpy.genfromtxt('TestingDataset.csv', delimiter=',', filling_values=0.0)
new_test_data = test_data_file[:,1:]

# prints
print "Train data shape", new_train_data.shape
print "Test data shape", new_test_data.shape


# In[7]:

# train the shit out of it
for e in range(epochs):
    for record in new_train_data:
        inputs = ((numpy.asfarray(record) * 0.90) + 0.01).reshape((50, 50))
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(record[0])] = 0.99
        nn.train(inputs, targets)

# test for whatever data we have
for record in new_test_data:
    inputs = ((numpy.asfarray(record)* 0.99) + 0.01).reshape(50, 50)
    outputs = nn.query(inputs)
    label = numpy.argmax(outputs)
    if label == None: #correct_label: # wtf is correct label?
        scorecard.append(1)
    else:
        scorecard.append(0)

# I don't know what it does LOL
scorecard_array = numpy.asarray(scorecard)

# I also don't know what it means. :(
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

