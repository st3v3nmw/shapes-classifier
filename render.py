import matplotlib.pyplot as plt
import numpy as np

f = open("epochs.txt", "r")
data = f.readlines()
for i in range(len(data)):
    data[i] = float(data[i][:-1])

y = np.array(data)

plt.plot(y)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.title("Change of accuracy over time")
plt.grid()
plt.show()
