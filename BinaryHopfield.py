import numpy as np
import numpy.random as rd

class BinaryHopfield(object):
    def __init__(self, n):
        self.n = n # no. of units
        self.V = rd.randint(0, 2, n) # activity vector of neurons
        self.T = rd.randn(n, n) # strength matrix of synaptic inputs
        self.T = (self.T + np.transpose(self.T)) / 2.0 # symmetric matrix with zero diagonals
        for i in range(n): self.T[i, i] = 0.0
    
    def __repr__(self):
        repr = f'V = {self.V}\nT = \n{self.T}'
        return repr
    
    def energy(self, I):
        return -1/2 * np.matmul(np.matmul(np.transpose(self.V), self.T), self.V) \
            - np.matmul(np.transpose(I), self.V)

    def step(self, input):
        I = np.array(input, dtype=np.float)
        assert(I.shape == (self.n,))

        # self.V = np.matmul(self.T, self.V) + I
        j = rd.randint(self.n)
        self.V[j] = 1 if np.sum(self.T[j,:] * self.V) + I[j] > 0.0 else 0

        # energy
        self.E = self.energy(I)

        return self.V, self.E

    def sim(self, inputs):
        inputs = np.array(inputs, dtype=np.float)
        assert(inputs.shape[1] == self.n)
        N = inputs.shape[0] # no. of simulaiton iteration

        Vs = np.zeros((N + 1, self.n), dtype=np.int)
        Vs[0, :] = self.V
        Es = np.zeros(N + 1, dtype=np.float)
        Es[0] = self.energy(np.zeros(self.n))

        for i in range(N):
            I = inputs[i,:]
            self.step(I)
            Vs[i + 1,:] = self.V
            Es[i + 1] = self.energy(I)

        return Vs, Es
