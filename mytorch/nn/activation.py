import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_norm = Z - np.max(Z, axis=self.dim, keepdims=True)
        sum_Z_norm = np.sum(np.exp(Z_norm), axis=self.dim, keepdims=True)
        self.A = np.exp(Z_norm) / sum_Z_norm
        # raise NotImplementedError
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.moveaxis(self.A, source=self.dim, destination=-1)
            # new_shape = self.A.shape
            self.A = self.A.reshape(-1, C)
            dLdA = np.moveaxis(dLdA, source=self.dim, destination=-1)
            dLdA = dLdA.reshape(-1, C)
        
        dLdZ = self.A * (dLdA - np.sum(dLdA * self.A, axis=1, keepdims=True))

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(shape)
            # self.A = np.moveaxis(self.A, source=-1, destination=self.dim)
            dLdZ = dLdZ.reshape(shape)
            # dLdZ = np.moveaxis(dLdZ, source=1, destination=self.dim)

        # raise NotImplementedError
        return dLdZ
 

    