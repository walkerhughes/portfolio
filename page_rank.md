---
layout: default
---

## Page Rank Algorithm For Setting up Your Brackets 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

```python 
import numpy as np 
import networkx as nx
from itertools import combinations
from scipy import linalg as la
```

```python 
class DiGraph:
    """
    A class for representing directed graphs via their adjacency matrices.

    PageRank vector can be computed using linearsolve, eigensolve, or iterrative solving methods. 
    """
    def __init__(self, A, labels = None): 
        """
        Modifies A so that there are no sinks in the corresponding graph,
        then calculates Ahat. Saves Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.n = len(A[0]) # will use more than once
        self.zeros, self.ones = np.zeros(self.n), np.ones(self.n)
        
        if not labels: # if no labels given make a range
            labels = list(range(self.n))
                
        if len(labels) != self.n: # if there are too many/too few
            raise ValueError("Labels not of correct dimension.")
        
        # eliminate sinks in A
        for i in range(self.n):
            if np.allclose(A[:, i], self.zeros):
                A[:, i] = self.ones
        
        # calculate A_hat with broadcasting along columns
        # store attributes
        self.Ahat = A / np.sum(A, axis = 0)
        self.labels = labels
```

We'll add the following methods to our class to find the eigenvector... 
```python 
    def linsolve(self, epsilon=0.85):
        """
        Computes the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # calculate vector of probabilities
        inv = la.inv(np.eye(self.n) - epsilon * self.Ahat)
        vec = ((1 - epsilon) / self.n) * self.ones
        p = inv @ vec
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p[i] for i in range(self.n)}

    def eigensolve(self, epsilon=0.85):
        """
        Computes the PageRank vector using the eigenvalue method.
        Normalizes the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # compute E matrix and coefficients
        E, coeff = np.ones((self.n, self.n)), ((1 - epsilon) / self.n)
        B = (epsilon * self.Ahat) + (coeff * E)
        # eigenvectors and values of B matrix
        eigs, vecs = la.eig(B)
        p = vecs[:, 0] / sum(vecs[:, 0])
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p[i] for i in range(self.n)}
        
        
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """
        Computes the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p0 = self.ones / self.n 
        
        # helper function to compute next p value
        next_p = lambda p0: (epsilon * np.dot(self.Ahat, p0)) + (1 - epsilon) * self.ones / self.n
            
        # iterate through
        for i in range(maxiter):
            p1 = next_p(p0)
            if max(np.abs(p1 - p0)) < tol:
                p0 = p1
                break
            p0 = p1
        
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p0[i] for i in range(self.n)}

```

```python 
A = np.array([[0, 0, 0, 0], 
              [1, 0, 1, 0], 
              [1, 0, 0, 1], 
              [1, 0, 1, 0]]) 

graph = DiGraph(A, labels = ["a", "b", "c", "d"])

print(graph.linsolve()) 
print(graph.eigensolve()) 
print(graph.itersolve()) 
```
```
{'a': 0.09575863576738085,'b': 0.2741582859641452,'c': 0.3559247923043289,'d': 0.2741582859641452}
{'a': 0.09575863576738085,'b': 0.2741582859641452,'c': 0.3559247923043289,'d': 0.2741582859641452}
{'a': 0.09575863576738085,'b': 0.2741582859641452,'c': 0.3559247923043289,'d': 0.2741582859641452}
```

