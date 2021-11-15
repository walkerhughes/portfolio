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

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    def __init__(self, A, labels = None):
        """
        Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.m, self.n = A.shape
        self.zeros, self.ones = np.zeros(self.n), np.ones(self.n)
        
        # if no labels given, make labels as numeric range 
        if not labels: 
            labels = list(range(self.n))
                
        # if there are too many/too few labels 
        if len(labels) != self.n:
            raise ValueError("Labels not of correct dimension.")
        
        # eliminate "sinks" in A 
        for i in range(self.n): 
            if np.allclose(A[:, i], self.zeros): 
                A[:, i] = self.ones
        
        # calculate A_hat with broadcasting along columns 
        self.Ahat = A / np.sum(A, axis = 0)
        self.labels = labels 
```

We'll add the following methods to our class to find the eigenvector... 
```python 
    def linsolve(self, epsilon = 0.85):
        """
        Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        inv = la.inv(np.eye(self.n) - (epsilon * self.Ahat))
        vec = ((1 - epsilon) / self.n) * self.ones
        p = np.dot(inv, vec) 
        
        return {self.labels[i]: p[i] for i in range(self.n)}

        
    def eigensolve(self, epsilon = 0.85):
        """
        Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        E, coeff = np.ones((self.n, self.n)), ((1 - epsilon) / self.n)
        B = (epsilon * self.Ahat) + (coeff * E)
        eigs, vecs = la.eig(B)
        p = vecs[:, 0] / sum(vecs[:, 0])
        
        return {self.labels[i]: p[i] for i in range(self.n)}
        
        
    def itersolve(self, epsilon = 0.85, maxiter = 100, tol = 1e-12):
        """
        Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p0 = self.ones / self.n 
        
        def next_p(p0): 
            return (epsilon * np.dot(self.Ahat, p0)) + ((1 - epsilon) * (self.ones / self.n))
        
        for i in range(maxiter): 
            p1 = next_p(p0)
            if max(np.abs(p1 - p0)) < tol: 
                p0 = p1
                break 
            p0 = p1 
        
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

```python 
def get_ranks(d):
    """
    Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # get labels and values
    labels, vals = list(d.keys()), list(d.values()) 
    
    # return list mapping labels sorted by rank
    return [labels[i] for i in np.argsort(vals)[:: -1]] 
```
```
print(get_ranks(d)) 
```
```
['c', 'd', 'b', 'a'] 
```



```python 
def rank_ncaa_teams(filename, epsilon = 0.85):
    """
    Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    # read in data, get unique set of teams, index them for easy accessability 
    with open(filename, "r") as myfile:
        data = myfile.read().strip()
        teams = sorted(set(data.replace("\n", ",").split(",")[2: ]))             
        team_indices = {team: i for i, team in enumerate(list(teams))}
    
    # init adjaacency matrix for DiGraph object 
    n = len(teams)
    A = np.zeros((n, n)) 
    
    # loop over individual game outcomes 
    for line in data.split("\n")[1: ]:

        # separate winner and loser from string 
        winner_loser = line.split(",")

        # get winner and loser from each game
        winner, loser = winner_loser[0], winner_loser[1] 

        # update adjacency matrix 
        row, col = team_indices[winner], team_indices[loser] 
        A[row, col] += 1
    
    # init directed graph, get page-rank team rakings
    graph = DiGraph(A, labels = sites)
    return get_ranks(graph.itersolve(epsilon = epsilon)) 
```
```
# top three teams going into March Madness 
rank_ncaa_teams(filename = "ncaa2010.csv", epsilon = 0.85)[: 3] 
```
```
['UConn', 'Kentucky', 'Louisville']
```

