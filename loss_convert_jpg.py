#plot loss
import matplotlib.pyplot as plt
import numpy as np
his = np.load("model_loss.npy")
plt.plot(his)
plt.title('Model loss')
plt.ylabel('loss')
plt.yscale("log")
plt.xlabel('epochs')
plt.legend(['training loss'], loc='upper left')
plt.savefig("Model_Loss.jpg")