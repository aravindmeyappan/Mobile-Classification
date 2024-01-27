import numpy as np

class neural_network():

    def __init__(
        self,
        layer_dims, 
        num_iterations=1000, 
        learning_rate=0.01, 
        beta1=0.9, 
        beta2=0.999, 
        epsilon=1e-8
    ):
        self.layer_dims = layer_dims
        self.num_iters = num_iterations
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameters = {}
        self.grads = {}
        self.v = {}
        self.s = {}
        self.L = len(self.layer_dims) # including the input layer as well
        self.m = None # number of examples
        self.t = 0 # time step for ADAM

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

    def batch_normalize(self, Z):
        """
        Helper function to compute the normalized value of Z of any layer
        
        Input: 
        the Z matrix corresponding to a specific layer 
        with dimensions (n_l,m). where n_l - number of units in layer l, m - number of training examples in training data

        Output:
        The normalized Z matrix with the same dimension as Z
        Mean and variance are computed using the Zs corresponding to different training examples (sample size is m)
        
        """
        mean = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        Z_normalized = (Z - mean) / np.sqrt(var + 1e-8)
        return Z_normalized, mean, var

    def linear_forward(self, A, W, b):
        """
        Helper function used to calculate the Z matrix of a layer l
        
        Inputs:
        A - activations from the previous layer. Dimension - (number of units in layer l-1, total number of examples)
        W - weights matrix of current layer. Dimension - (number of units in layer l, number of units in layer l-1)
        b - bias of current layer. Dimension - (number of units in layer l, 1)

        Outputs:
        The Z matrix along with all the inputs stored in cache useful in back propogation
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def batch_normalize_forward(self, Z, gamma, beta):
        """
        Helper function to calculate the Z tilde to be fed to the activation function before feeding to the next layer

        Inputs:
        Z - The Z matrix computed from linear_forward function
        gamma - the parameter gamma associated with normalization. Dimension - (number of units in current layer, 1)
        beta - the parameter associated with normalization. Dimension - (number of units in current layer, 1)

        Note:
        The parameter beta is not to be mistaken with the hyper parameters beta1 and beta2

        Outputs:
        Z_tilde - the scaled version of Z_normalized obtained from batch_normalize
        """
        Z_normalized, mean, var = self.batch_normalize(Z)
        Z_tilde = gamma * Z_normalized + beta # this is element wise multiplication 
        cache = (Z, Z_normalized, mean, var, gamma, beta)
        return Z_tilde, cache

    def activation_forward(self, Z, activation):
        """
        Helper function to compute the activation of Z_tilde obtained from previous function.
        Applied the activation based on the choice.
        """
        if activation == "relu":
            A = self.relu(Z)
        elif activation =="tanh":
            A = self.tanh(Z)
        elif activation == "softmax":
            A = self.softmax(Z)
    
        cache = Z
        return A, cache

    def initialize_parameters_deep(self):
        """
        Initializing the weights using xavier's initialization(for ReLU activation) 
        for all the layers according the layer dimension info given by layer_dims

        Dimensions of parameters used in forward propogation:
        Weights - (number of units in current layer, number of units in prev layer)
        b - (number of units in current layer, 1)

        Dimensions of parameters used in batch normalization:
        gamma - (number of units in current layer, 1)
        beta - (number of units in current layer, 1)
        """
    
        for l in range(1, self.L):
            # following the logic that the dimension of the weights of any layer will be 
            # (number of units in current layer, number of units in prev layer)
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])

            # b, gamma and beta parameters all will will have a dimension that depends on the number of units in the current layer
            # given by (number of units in current layer, 1)
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
            self.parameters[f'gamma{l}'] = np.ones((self.layer_dims[l], 1))
            self.parameters[f'beta{l}'] = np.zeros((self.layer_dims[l], 1))
    
        return

    def forward_propagation_deep(self, X, activations):
        """

        Inputs:
        Activations - this is a list of size (number_layers - 1) L which basically tells about the activation to use for each layer
        for eg. ['relu', 'relu', 'softmax'] means that the first 2 hidden layers will have relu activation and the
        output layer will have softmax activation
        """
        caches = []
        A = X
        L = len(self.parameters) // 4
       
        for l in range(1, L):
            # A_prev gets updated in each iteration as A keeps getting updated in each iteration
            A_prev = A
            
            # the linear_forward function returns Z, (A_prev, W[l], b[l]) stored in cache
            Z, linear_cache = self.linear_forward(A_prev, self.parameters[f'W{l}'], self.parameters[f'b{l}'])
            
            # the batch_normalize_forward returns Z_tilde, (Z, Z_normalized, mean, var, gamma, beta) stored in cache
            Z_tilde, batch_cache = self.batch_normalize_forward(Z, self.parameters[f'gamma{l}'], self.parameters[f'beta{l}'])
            
            # the activation_forward returns A, (Z) stored in cache
            A, activation_cache = self.activation_forward(Z_tilde, "relu")

            # storing all types of cache to one variable
            cache = (linear_cache, batch_cache, activation_cache)

            # appending that to the caches list. caches will contain the 3 types of cache for all the layers
            # the ith item in caches list will have all the cache pertaining to the ith layer in the network
            caches.append(cache)
    
        # Last layer (softmax activation)
        ZL, linear_cache = self.linear_forward(A, self.parameters[f'W{L}'], self.parameters[f'b{L}'])
        ZL_tilde, batch_cache = self.batch_normalize_forward(ZL, self.parameters[f'gamma{L}'], self.parameters[f'beta{L}'])
        AL, activation_cache = self.activation_forward(ZL_tilde, "softmax")
        cache = (linear_cache, batch_cache, activation_cache)
        caches.append(cache)
    
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Helper function to compute the cost function that compares the activations of the final layer and the target labels.
        Assumes that the target labels Y has a dimension (m,) where m denotes the number of examples
        """
        Y_one_hot = np.eye(AL.shape[0])[Y.astype(int)].T.squeeze()
        cost = -1/self.m * np.sum(Y_one_hot * np.log(AL+self.epsilon))
        return cost

    def activation_backward(self, dA, cache, activation):
        """
        Helper function to calculate the gradients of Z_tilde because we get activations of any layer from Z_tilde
        So in backward propogation we get dZ_tilde from dA

        Inputs:
        dA - derivative of the cost function with respect to the activation of that layer 
        so for the last layer L it'll be dcost/dAL

        Output:
        dZ - which is actually dZ_tilde. we get the derivative of the cost function with respect to 
        Z_tilde of that layer. so for the last layer L it'll be dcost/dZL_tilde = (dcost/dAL)*(dAL/dZL_tilde) by chain rule
        """
        Z = cache
        if activation == "relu":
            dZ = dA * self.relu_derivative(Z)
        elif activation == "tanh":
            dZ = dA * self.tanh_derivative(Z)
        elif activation == "softmax":
            dZ = dA
        return dZ

    def batch_normalize_backward(self, dZ_tilde, cache):
        """
        Helper function to compute the gradients of Z, gamma, beta because in forward propogation we get Z_tilde from Z, gamma, beta
        so in backward propogation we get dZ, dGamma, dBeta from dZ_tilde

        Inputs:
        dZ_tilde - derivative of the cost function with respect to Z_tilde of that layer. 
        so for the last layer it will be dcost/dZL_tilde

        Output:
        dZ - derivative of the cost function with respect to Z of that layer.
        so for last layer it will be dcost/dZL = (dcost/dZL_tilde)*(dZL_tilde/dZL)
        """
        Z, Z_normalized, mean, var, gamma, beta = cache
    
        dZ_normalized = dZ_tilde * gamma # dZL_tilde/dZL = gamma 
        dVar = np.sum(dZ_normalized * (Z - mean), axis=1, keepdims=True) * -0.5 * (var + 1e-8)**(-1.5)
        dMean = np.sum(dZ_normalized, axis=1, keepdims=True) * -1 / np.sqrt(var + 1e-8)
        
        dZ = (dZ_normalized / np.sqrt(var + 1e-8)) + (dVar * 2 * (Z - mean) / self.m) + (dMean / self.m)
        dGamma = np.sum(dZ_tilde * Z_normalized, axis=1, keepdims=True)
        dBeta = np.sum(dZ_tilde, axis=1, keepdims=True)
    
        return dZ, dGamma, dBeta

    def linear_backward(self, dZ, cache):
        """
        Helper function to compute the gradients of A_prev, W, b because in forward propogation we get Z from A_prev, W, b
        so in backward propogation we get dA_prev, dW, db from dZ

        Inputs:
        dZ - derivative of the cost function with respect to Z of that layer. 
        so for the last layer it will be dcost/dZL

        Output:
        dA_prev, dW, db - derivative of the cost function with respect to W, b of that layer and A of previous layer.
        so for last layer it will be dcost/dWL = (dcost/dZL)*(dZL/dWL) and so on for all 3 of them
        """
        A_prev, W, b = cache
        dW = 1/self.m * np.dot(dZ, A_prev.T)
        db = 1/self.m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
        return dA_prev, dW, db

    def backward_propagation_deep(self, AL, Y, caches, activations):
        L = len(caches)
    
        # Convert Y to one-hot encoded matrix
        Y_one_hot = np.eye(AL.shape[0])[Y.astype(int)].T.squeeze()
    
        # Compute gradient of the cost with respect to AL for softmax activation
        dAL = AL - Y_one_hot
        self.grads['dAL'] = dAL
        
        # Last layer (softmax activation)
        current_cache = caches[L-1]
        linear_cache, batch_cache, activation_cache = current_cache
        dZL_tilde = self.activation_backward(dAL, activation_cache, "softmax")
        dZL, dGammaL, dBetaL = self.batch_normalize_backward(dZL_tilde, batch_cache)
        dA_prev, dW, db = self.linear_backward(dZL, linear_cache)
        self.grads[f'dA{L-1}'], self.grads[f'dW{L}'], self.grads[f'db{L}'] = dA_prev, dW, db
        self.grads[f'dGamma{L}'], self.grads[f'dBeta{L}'] = dGammaL, dBetaL
    
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            linear_cache, batch_cache, activation_cache = current_cache
            dZ_tilde = self.activation_backward(self.grads[f'dA{l+1}'], activation_cache, "relu")
            dZ, dGamma, dBeta = self.batch_normalize_backward(dZ_tilde, batch_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            self.grads[f'dA{l}'], self.grads[f'dW{l+1}'], self.grads[f'db{l+1}'] = dA_prev, dW, db
            self.grads[f'dGamma{l+1}'], self.grads[f'dBeta{l+1}'] = dGamma, dBeta
    
        return 

    def initialize_adam(self):
        L = len(self.parameters) // 4  # Considering gamma and beta for each layer
        
        for l in range(1, L+1):
            self.v[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.v[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            self.v[f'dGamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
            self.v[f'dBeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
    
            self.s[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.s[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            self.s[f'dGamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
            self.s[f'dBeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
    
        return 

    def update_parameters_adam(self):
        
        L = len(self.parameters) // 4  # Considering gamma and beta for each layer
    
        for l in range(1, L+1):
            self.v[f'dW{l}'] = self.beta1 * self.v[f'dW{l}'] + (1 - self.beta1) * self.grads[f'dW{l}']
            self.v[f'db{l}'] = self.beta1 * self.v[f'db{l}'] + (1 - self.beta1) * self.grads[f'db{l}']
            self.v[f'dGamma{l}'] = self.beta1 * self.v[f'dGamma{l}'] + (1 - self.beta1) * self.grads[f'dGamma{l}']
            self.v[f'dBeta{l}'] = self.beta1 * self.v[f'dBeta{l}'] + (1 - self.beta1) * self.grads[f'dBeta{l}']
    
            self.s[f'dW{l}'] = self.beta2 * self.s[f'dW{l}'] + (1 - self.beta2) * (self.grads[f'dW{l}']**2)
            self.s[f'db{l}'] = self.beta2 * self.s[f'db{l}'] + (1 - self.beta2) * (self.grads[f'db{l}']**2)
            self.s[f'dGamma{l}'] = self.beta2 * self.s[f'dGamma{l}'] + (1 - self.beta2) * (self.grads[f'dGamma{l}']**2)
            self.s[f'dBeta{l}'] = self.beta2 * self.s[f'dBeta{l}'] + (1 - self.beta2) * (self.grads[f'dBeta{l}']**2)
    
            v_corrected_dW = self.v[f'dW{l}'] / (1 - self.beta1**self.t)
            v_corrected_db = self.v[f'db{l}'] / (1 - self.beta1**self.t)
            v_corrected_dGamma = self.v[f'dGamma{l}'] / (1 - self.beta1**self.t)
            v_corrected_dBeta = self.v[f'dBeta{l}'] / (1 - self.beta1**self.t)
    
            s_corrected_dW = self.s[f'dW{l}'] / (1 - self.beta2**self.t)
            s_corrected_db = self.s[f'db{l}'] / (1 - self.beta2**self.t)
            s_corrected_dGamma = self.s[f'dGamma{l}'] / (1 - self.beta2**self.t)
            s_corrected_dBeta = self.s[f'dBeta{l}'] / (1 - self.beta2**self.t)
    
            self.parameters[f'W{l}'] -= self.learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW) + self.epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + self.epsilon)
            self.parameters[f'gamma{l}'] -= self.learning_rate * v_corrected_dGamma / (np.sqrt(s_corrected_dGamma) + self.epsilon)
            self.parameters[f'beta{l}'] -= self.learning_rate * v_corrected_dBeta / (np.sqrt(s_corrected_dBeta) + self.epsilon)

        return 
    
    def fit(self, X, Y):
        activations = ["relu"] * (len(self.layer_dims) - 2) + ["softmax"]

        self.n, self.m = X.shape

        self.initialize_parameters_deep()
        self.initialize_adam()

        for i in range(self.num_iters):
            AL, caches = self.forward_propagation_deep(X, activations)
            cost = self.compute_cost(AL, Y)
            self.backward_propagation_deep(AL, Y, caches, activations)
    
            # Update parameters with Adam
            self.t += 1
            self.update_parameters_adam()
    
            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')
        return "Fitting Complete"

    def predict_deep(self, X):
        activations = ["relu"] * (len(self.layer_dims) - 2) + ["softmax"]
        AL, _ = self.forward_propagation_deep(X, activations)
        predictions = np.argmax(AL, axis=0)
        return predictions
            