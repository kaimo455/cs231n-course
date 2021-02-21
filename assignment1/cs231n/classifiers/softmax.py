from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    for i in range(num_samples):
      scores = np.dot(X[i], W)
      scores -= np.max(scores)  # avoid numerical explosion
      prob = np.exp(scores) / np.sum(np.exp(scores))
      loss -= np.log(prob[y[i]])

      dscores = prob.copy()
      dscores[y[i]] -= 1
      dW += np.dot(X[i].T[:, np.newaxis], dscores[np.newaxis, :])

    loss /= num_samples
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_samples
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    # forward
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    loss -= np.mean(np.log(prob[np.arange(num_samples), y]), axis=0)
    loss += 0.5 * reg * np.sum(W * W)
    # backward
    dscores = prob.copy()
    dscores[np.arange(num_samples), y] -= 1
    dW = np.dot(X.T, dscores)
    dW /= num_samples
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
