from Perceptron import Perceptron
import numpy as np

perc = Perceptron(0.5)

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

and_func = np.array([-1, -1, -1, 1])
or_func = np.array([-1, 1, 1, 1])
xor_func = np.array([1, -1, -1, -1])
xx_func = np.array([1, 1, 1, -1])

perc.train(x, xx_func)
print(perc.predict(np.array([0, 0])))
print(perc.predict(np.array([0, 1])))
print(perc.predict(np.array([1, 0])))
print(perc.predict(np.array([1, 1])))
print(perc.w)
