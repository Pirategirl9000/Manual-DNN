import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset(rand):
    np.random.seed(rand)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

#___________________________________________________________________________

X, Y = load_planar_dataset(1)

plt.scatter(X[0, :], X[1, :], c = Y, s=40, cmap=plt.cm.Spectral)

w1 = np.random.randn(4, X.shape[0]) * 0.01
b1 = np.zeros(shape = (4, 1))

w2 = np.random.randn(Y.shape[0], 4) * 0.01
b2 = np.zeros(shape = (Y.shape[0], 1))

def forward_propagation(X, w1, w2, b1, b2):
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    #Cache values
    cache = {
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2
    }

    return a2, cache

def compute_cost(a2, Y):
    m = Y.shape[1]

    cost_sum = np.multiply(np.log(a2), Y) + np.multiply((1-Y), np.log(1-a2))
    cost = np.sum(cost_sum) / m
    cost = np.squeeze(cost)

    return cost

def back_propagation(X, Y, w1, b1, w2, b2, cache, learning_rate):
    a1 = cache['a1']
    a2 = cache['a2']
    m = Y.shape[1]

    dz2 = a2 - Y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    #Get new weights/biases
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    return w1, w2, b1, b2

for i in range(0, 1000):
    az, cache = forward_propagation(X, w1, w2, b1, b2)

    cost = compute_cost(az, Y)

    w1, w2, b1, b2 = back_propagation(X, Y, w1, b1, w2, b2, cache, 0.01)
