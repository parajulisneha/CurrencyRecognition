import numpy
import sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Need 0.18.2 version, for the NN
print('The scikit-learn version is {}.'.format(sklearn.__version__))

mapping = {
    0: "5 rupees",
    1: "10 rupees",
    2: "20 rupees",
    3: "50 rupees",
    4: "100 rupees",
    5: "500 rupees",
    6: "1000 rupees",
}

training_data_file = numpy.genfromtxt('TrainingDataset.csv', delimiter=',', filling_values=0.0)

# X indicates input train value
X_Train = training_data_file[:, 1:-5]

# Y indicates target value
Y_Train = training_data_file[:,0]
print "X Train", X_Train.shape
print "Y Train", Y_Train.shape

test_data_file = numpy.genfromtxt('TestingDataset.csv', delimiter=',', filling_values=0.0)
X_Test =  test_data_file[:,1:]
Y_Test =  test_data_file[:,0]
print "X Test", X_Test.shape
print "Y Test", Y_Test.shape

# Defining the model. MOST IMPORTANT PART HO HAI. We can save it later as well

# MLP means: Multi-layer Perceptron classifier. See: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', max_iter=3000,
                    hidden_layer_sizes=(50,), random_state=1)

# training the shit out of it :D
mlp.fit(X_Train, Y_Train)

# predicting the output. All done by library LOL ;)
predictions = mlp.predict(X_Test)
print predictions

for each in predictions:
    print mapping[each]


# this is confusion_matrix. needs time to understand :D hehehe
print(confusion_matrix(Y_Test,predictions))

# lol I don't know what the hell is this :p
print(classification_report(Y_Test,predictions))