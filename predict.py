# Mwangi Stephen, Mutuku Moses, Mokaya Sharon
from neuralnetwork import NeuralNetwork
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys

print("SHAPE CLASSIFIER")
print("=================")

filename = sys.argv[1]
pca = PCA(n_components=5)
nn = NeuralNetwork.load_from_file("models/model0.json")
classes = {0: "Triangle", 1: "Square", 2: "Circle"}

# read image from file
image = plt.imread(filename)
# perform Principal Component Analysis
data = pca.fit_transform(image).ravel()
# pass the processed image to the neural network and obtain a classification
results = nn.predict(data)

# print the results
print("Prediction for %s: %s" % (filename, classes[int(np.argmax(results))]))
msg = "RESULT: This is not a triangle: {type}".format(type=classes[int(np.argmax(results))])
if not np.argmax(results):
    msg = "RESULT: This is a triangle"

print("\nRAW PREDICTIONS")
print("=================")
for i in range(len(results)):
    print("%s: %.4f" % (classes[i], results[i]))
print()
# draw image
plt.imshow(image, cmap='gray')
plt.title(msg)
plt.show()
