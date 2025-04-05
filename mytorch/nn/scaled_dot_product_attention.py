import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        K_T = np.moveaxis(K, source=-2, destination=-1)  # from (N, ..., H, S, E) to (N, ..., H, E, S)
        self.d_k = Q.shape[-1]
        self.Q, self.K, self.V = Q, K, V
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        scaled_dot_product = Q @ K_T / np.sqrt(self.d_k)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product -= mask * self.eps

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores @ V

        # Return output
        # raise NotImplementedError
        return output

    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass
        A_T = np.moveaxis(self.attention_scores, source=-2, destination=-1)  # (N, ..., H, S, L)
        V_T = np.moveaxis(self.V, source=-2, destination=-1)  # (N, ..., H, Ev, S)

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = A_T @ d_output
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = d_output @ V_T
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product /= np.sqrt(self.d_k)
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K
        # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.moveaxis(d_scaled_dot_product, source=-2, destination=-1) @ self.Q
        
        # Return gradients for Q, K, V
        # raise NotImplementedError
        return d_Q, d_K, d_V
