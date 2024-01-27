import numpy as np

class Logistic_NN():
    """
    A slightly modified logistic regression with softmax activation on the output instead of sigmoid due to the presence of multiple classes
    can be modified to perform multiple one v rest classifications and accumulating their results by using sigmoid activation for each output
    
    Input: 
    X - feature data of the dimension (num_features, num_samples)
    Y - target labels in 1D array form, highlighting the class. dimension (num_samples, )
    """

    def __init__(self, num_classes, learning_rate=0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.iterations = num_iters
        self.num_classes = num_classes

    def softmax(self, z):
        # helper function - computes the softmax activation of the linear output (w.T . X)
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def initialize_parameters(self, n, num_classes):
        """
        Helper function to assist in initializing the parameters needed for logistic regression
        """
        w = np.random.randn(n, num_classes) * 0.01
        b = np.zeros((num_classes, 1))
        return w, b

    def propagate(self, X, Y):
        """
        Performs both forward propogation and the backward propogation of the algorithm and returns the gradients 
        """
        # Forward propagation
        Z = np.dot(self.w.T, X) + self.b
        A = self.softmax(Z)
        cost = -1/self.m * np.sum(np.log(A[Y, np.arange(self.m)]))
        
        # Backward propagation - 
        # this method of calculating dz is specific for the case where we use softmax activation instead of sigmoid
        # in sigmoid we simply do dz = a-y, but here since a is one hot encoded and y is not we do this
        dz = A.copy()
        dz[Y, np.arange(self.m)] -= 1
        dw = 1/self.m * np.dot(X, dz.T)
        db = 1/self.m * np.sum(dz, axis=1, keepdims=True)
        
        grads = {"dw": dw, "db": db}
        
        return grads, cost

    def fit(self, X, Y):
        """
        The fit method allows the model to update its parameters with respect to the data and finally return the parameters 
        after a certain number of specified passes. 
        """
        self.n,self.m = X.shape

        # initialize the parameters 
        self.w, self.b = self.initialize_parameters(self.n, self.num_classes)

        # use the propogate function to update the parameters
        for i in range(self.iterations):
            grads, cost = self.propagate(X, Y)
            
            # Update parameters
            self.w -= self.learning_rate * grads["dw"]
            self.b -= self.learning_rate * grads["db"]
    
            if i%200==0:
                print(f"cost after {i} iterations: {cost}")
        return self.w, self.b

    def predict(self, X):
        Z = np.dot(self.w.T, X) + self.b
        A = self.softmax(Z)
        predictions = np.argmax(A, axis=0)
        return predictions

    def accuracy(self, predictions, actual_labels):
        correct_predictions = np.sum(predictions == actual_labels)
        total_examples = len(actual_labels[0])
        acc = correct_predictions / total_examples
        return acc
    

class NN_2_layer():

    def __init__(self, num_classes, num_hidden, learning_rate = 0.01, num_iters = 1000):
        self.num_classes = num_classes
        self.num_hidden = num_hidden # number of hidden units in the 1 hidden layer
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.params = {}

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def initialize_parameters(self):
        self.params['w1'] = np.random.randn(self.num_hidden, self.n) * np.sqrt(2 / self.n)  # Xavier initialization
        self.params['b1'] = np.zeros((self.num_hidden, 1))
        self.params['w2'] = np.random.randn(self.num_classes, self.num_hidden) * np.sqrt(2 / self.num_hidden)  # Xavier initialization
        self.params['b2'] = np.zeros((self.num_classes, 1))
        return

    def forward_propagation(self, X, Y):
        Z1 = np.dot(self.params['w1'], X) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(self.params['w2'], A1) + self.params['b2']
        A2 = self.softmax(Z2)
        

        Y_one_hot = np.eye(self.num_classes)[Y].T
        Y_one_hot = Y_one_hot.reshape(A2.shape) # Ensure shapes are compatible
       
        # avoid numerical instability
        epsilon = 1e-15
        A2 = np.maximum(epsilon, A2)
        cost = -np.sum(Y_one_hot * np.log(A2)) / self.m
        
        return A1, A2, cost

    def backward_propagation(self, A1, A2, X, Y):

        # this is a way to calculate the dz for softmax activation since the output is similar to a one hot encoded label
        dz2 = A2.copy()
        dz2[Y, np.arange(self.m)] -= 1
        
        dw2 = 1/self.m * np.dot(dz2, A1.T)
        db2 = 1/self.m * np.sum(dz2, axis=1, keepdims=True)
        
        dz1 = np.dot(self.params['w2'].T, dz2) * self.relu_derivative(A1)
        dw1 = 1/self.m * np.dot(dz1, X.T)
        db1 = 1/self.m * np.sum(dz1, axis=1, keepdims=True)
        
        grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
        
        return grads

    def update_parameters(self, grads):
        self.params['w1'] -= self.learning_rate * grads["dw1"]
        self.params['b1'] -= self.learning_rate * grads["db1"]
        self.params['w2'] -= self.learning_rate * grads["dw2"]
        self.params['b2'] -= self.learning_rate * grads["db2"]
        return

    def fit(self, x, y):

        self.n, self.m = x.shape

        self.initialize_parameters() # just call the function and it stores the initialized parameters in the params attribute

        for i in range(self.num_iters):
            A1, A2, cost = self.forward_propagation(x, y)
            grads = self.backward_propagation(A1, A2, x, y)
            self.update_parameters(grads) # just calling the function will update the parameters stored in the params attribute
    
            if i%1000==0:
                print(f"cost after {i} iterations: {cost}")

    
    def predict(self, X, Y):
        _, A2, _ = self.forward_propagation(X,Y)
        predictions = np.argmax(A2, axis=0) # converting the predictions in the form of the initial target label format
        correct = np.sum(predictions == Y)
        print(f"Accuracy: {correct/len(Y[0])}")
        return predictions