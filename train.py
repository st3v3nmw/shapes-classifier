# Mwangi Stephen, Mutuku Moses, Mokaya Sharon
from neuralnetwork import NeuralNetwork
from sklearn.decomposition import PCA
from matplotlib.pyplot import imread
import glob

print("SHAPE CLASSIFIER")
print("=================")

training_path = 'dataset/train'
test_path = 'dataset/test'
paths = ['triangles', 'squares', 'circles']
pca = PCA(n_components=5)

train_data, test_data = [], []
labels, test_labels = [], []
classes = {"triangles": [0.99, 0.01, 0.01], "squares": [0.01, 0.99, 0.01], "circles": [0.01, 0.01, 0.99]}


def load_training_data():
    # function to load the training data
    print("Importing training data...")
    for name in paths:
        for image_path in glob.glob("{path}/{name}/*.png".format(path=training_path, name=name)):
            image = imread(image_path)
            train_data.append(image)
            labels.append(classes[name])


def load_test_data():
    # function to load the test data
    print("Importing testing data...")
    for name in paths:
        for image_path in glob.glob("{path}/{name}/*.png".format(path=test_path, name=name)):
            image = imread(image_path)
            test_data.append(image)
            test_labels.append(classes[name])


load_training_data()
load_test_data()

print("Performing Principal Component Analysis...")
# Perform Principal Component Analysis on the train and test data
for j in range(len(train_data)):
    train_data[j] = pca.fit_transform(train_data[j]).ravel()
for j in range(len(test_data)):
    test_data[j] = pca.fit_transform(test_data[j]).ravel()

accuracy = 0
# create an instance of the network
nn = NeuralNetwork(1000, 500, 3, 0.01)

print("Training...")
# iterate N times perform training at each step
for epoch in range(100):
    for j in range(len(train_data)):
        nn.train(train_data[j], labels[j])
    accuracy = nn.accuracy(test_data, test_labels)
    print("\tepoch {i}, accuracy = {accuracy}".format(i=epoch+1, accuracy=accuracy))

# ask the user if they'd like to save the trained model, then the appropriate action is taken
save = input("Save current model with accuracy %.4f? [Y/N]: " % accuracy)
if save == "Y":
    model_file_path = "models/model.json"
    nn.write_to_file(model_file_path)
    print("Model saved to %s" % model_file_path)
else:
    print("Model not saved.")
print("Done")
