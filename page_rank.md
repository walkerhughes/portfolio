---
layout: default
---

## Page Rank Algorithm For Setting up Your Brackets 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

```python 
import numpy as np 
from itertools import combinations
from scipy import linalg as la
```

### Initial Motivation 
Many real-world systems can be modeled as networks and codified as directed graphs. These are easily storable as arrays in python, and the PageRank Algorithm is a way to rank the nodes in these networks by importance. Here I implement the PageRank Algorithm to rank NCAA basketball teams going into March Madness on the basis of the teams they each played against and how many times they won or lost against each team. 

We’ll take the following approach, representing the outcomes of these games as an adjacency matrix A. This will list all of the N teams we are concerned with and numerically represent these pairings as an N x N array where the columns and rows indicate a specific team. As an example, for any two teams i and j, node-i,j in our adjacency matrix will represent the number of times that team i beat team j prior to March Madness as an integer value. Likewise, node-j,i in the adjacency matrix  will be the number of times that team j beat team i prior to March Madness. To fix the problem of “sinks” in the matrix. These are locations where a team did not play against any of the other teams in the NCAA. For example, for team i, node-i,i will be a 0, which is mathematically undesirable for our algorithm. We’ll replace any column of row where this occurs with a column or row of all ones. 

We can extend our use of a simple adjacency matrix though to incorporate the relative amount of wins each team had against any other.  This is done by finding the percentage of a team’s  total wins represented by their wins against each of their components. This is meaningful since knowing if a team i is relatively more likely to win against another team j does not rely on the total number of games a team played in a season, and not all teams in our data played the same number of games.

I implement a class called ```DiGraph``` that accepts an Adjacency Matrix A, eliminates any "sinks" init, and then finds the relative frequency of a team's wins against any other team (stored as A-hat). We can then run the PageRank Algorithm on A-hat. 


```python 
class DiGraph:
    """
    A class for representing directed graphs via their adjacency matrices.
    """
    def __init__(self, A, labels = None):
        """
        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.

            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.m, self.n = A.shape
        self.zeros, self.ones = np.zeros(self.n), np.ones(self.n) 
        
        # eliminate "sinks" in A  
        for i in range(self.n): 
            if np.allclose(A[:, i], self.zeros): 
                A[:, i] = self.ones
        
        # calculate A-hat with broadcasting along columns 
        self.Ahat = A / np.sum(A, axis = 0)
        self.labels = labels 
```
### PageRank Vector 

The PageRank vector is essentially just a steady-state vector to the markov process implied by our adjacency matrix A-hat. While this is easily found for simple, low-dimensional systems, more sophisticated versions to find the PageRank vector exist when we are dealing with higher-dimensioinality. 

We'll add the following methods to our class to find the PageRank vector in 3 different forms then confirm they all agree with each other on a simple test case. I'll only give an intuitive appeal for how the different methods work and won't go too into the math. 

#### Linear Solver 
This method essentially finds the probability limit of being in state i (node i) as time approaches infinity. Translated to basketball, this is attempting to find the probability of being the winning team between any two pairings if many many games were played.  

#### Eigen-Solver
This is essentially the power method for solving a linear system, then normalizing the resulting vector. 

#### Iterative Solving 
This method works similarly to the Eigen-Solver method but has a simple tolerance cut-off for when we ought to stop iterating. This relies on finding a sequence of potential pagerank vectors, and stopping either when we've iterated enough times, or when the norm between two sequential vectors is smaller than our indicated tolerance (this is esentially saying that we've either tried hard enough or are close enough to an adequate steady-state vector)

These methods are implemented below. 


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
# a simple test case to make sure our three page rank methods aagree 
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

Since these methods all agree, we can move onto actually ranking nodes based on this vector. This is really the key part of the ranking system, since it translates our steady-state vector into an actionable label. It's very simple: sort the labels associated with the steady-state vector acording to their numeric value in decreasing order.  

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

Now for the fun stuff. The file ```ncaa2010.csv``` contains the winners and losers from NCAA basketball games in 2010 before March Madness was in full swing. 


```python 
def rank_ncaa_teams(filename = "ncaa2010.csv", epsilon = 0.85):
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
    
    # init adjacency matrix for DiGraph object 
    n = len(teams)
    A = np.zeros((n, n)) 
    
    # loop over individual game outcomes 
    for line in data.split("\n")[1: ]:

        # separate winner and loser from string 
        winner_loser = line.split(",")

        # get winner and loser from each game
        winner, loser = winner_loser[0], winner_loser[1] 

        # update adjacency matrix 
        winner_row, loser_col = team_indices[winner], team_indices[loser] 
        A[winner_row, loser_col] += 1 
    
    # init directed graph, get page-rank team rakings
    graph = DiGraph(A, labels = teams)
    return get_ranks(graph.itersolve(epsilon = epsilon)) 
```
```
# top three teams going into March Madness 
rank_ncaa_teams(filename = "ncaa2010.csv", epsilon = 0.85)[: 3] 
```
```
['UConn', 'Kentucky', 'Louisville']
```

