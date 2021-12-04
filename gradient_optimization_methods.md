---
layout: default
---

## Optimization though Gradient Methods 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>


```python
import numpy as np 
import scipy.optimize as opt 
from scipy.linalg import norm 
import matplotlib.pyplot as plt 
```

Vanilla gradient descent methods can easily be improved when an appropriate search direction and step size are chosen during each iteration of optimization. Usually, the search direction is the negative gradient of the function we seek to optimize, since this points in the direction of greatest descent to the minimizer we are seeking. In practice though this can be slow to converge since computing high dimensional derivatives can be computationally complex. 

The conjugate gradient algorithm uses a similar method but can converge much faster. I'll implement both a steepest descent and a conjugate gradient routine here. 


### Steepest Descent Method 

For this iterative method, we proceed similarly to vanilla gradient descent with an initial guess for the minimizer x0, but at each step we find an appropriate step size "alpha' for the following intermediate minimizer
<p><span class="math display">\[x_{k+1} = x_k - \alpha_k Df(X_k)^T\]</span></p>
where Df is the derivative of f and 
<p><span class="math display">\[\alpha_k = argmin_{\alpha} f(x_k - \alpha Df(x_k)^T) \]</span></p>


In essence, we have to solve an intermediate optimization problem for the step size at each iteration of this method to find the minimizer. 


```python
def steepest_descent(f, Df, x0, tol = 1e-5, maxiter = 100):
    """
    Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    for i in range(maxiter): 
        
        dfx = Df(x0) 

        # define line-search function for alpha
        opt_alpha = lambda a: f(x0 - (a * dfx))  

        # update x0 with optimal alpha
        x0 -= (opt.minimize_scalar(opt_alpha).x * dfx) 
        
        if norm(Df(x0), np.inf) < tol:  
            return x0, True, i + 1 
    
    # return if no convergence
    return x0, False, i + 1
```

We'll test this on a pretty easy function whose minimizer if the zero vector to check for convergence. Our optimization problem then becomes the following

<p><span class="math display">\[min_{x, y, z} f(x, y, z) = x^4 + y^4 + z^4 \]</span></p>


```python
f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
Df = lambda x: np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
x0 = np.array([1, 1, 1])

x, t, n = steepest_descent(f, Df, x0)
print("Convergence: {}\n\nConverged to {} in {} iteration(s)".format(t, x, n))
print("All close to [0, 0, 0] ? --> {}\n".format(np.allclose(x, np.zeros(3))))
```

```
Convergence: True

Converged to [9.24407773e-10 9.24407773e-10 9.24407773e-10] in 1 iteration(s)
All close to [0, 0, 0] ? --> True
```

### Conjugate Gradient Method 


The steepest descent method isn't always the best though, and can often get stuck in local minimums and flat areas. The conjugate gradient method, on the other hand, chooses a search direction that is guaranteed to be a direction of descent (thus avoiding getting stuck in a flat area), but not necessarily the steepest descent direction. 


The name 'conjugate" comes from a specific set of vectors {x_1, ... x_n} used in these routines that diagonalize the matrix Q when solving a linear system of the form Qx = b. Using this routines also guarantees that an iterative method to solve this system willl only require as many iterations as there are vectors {x_1, ... x_n}. 

Again, we'll implement this method and test it on a simple linear system. We will minimize the following function 

<p><span class="math display">\[f(x, y) = x^2 + 2y^2 - x - 8y \]</span></p>

which can be reworked to the following matrix form

<p><span class="math display">\[Q = 
\begin{bmatrix}
2 & 0 \\
0 & 4
\end{bmatrix}, b = \begin{bmatrix}
1 \\
8 
\end{bmatrix}\]</span></p> 

This system has a minimizer at [0.5, 2]


```python
def conjugate_gradient(Q, b, x0, tol = 1e-4):
    """
    Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x, convergence = x0, False 
    r0 = np.dot(Q, x) - b
    
    d0, n, k = -r0, len(b), 0
    
    while k < n: 
        
        # define our alpha for the system 
        alpha = np.dot(r0.T, r0) / np.dot(d0.T, np.dot(Q, d0)) 
        x = x + (alpha * d0) 
        r1 = r0 + (alpha * np.dot(Q, d0))
        
        beta = np.dot(r1.T, r1) / np.dot(r0.T, r0)
        d0 = -r1 + (beta * d0) 
        
        r0 = r1 
        k += 1
        
        if norm(r0) < tol: 
            convergence = True
            break 
        
    return x, convergence, k
```

```python
Q, b = np.array([[2, 0], [0, 4]]), np.array([1, 8])
x0 = np.array([0, 0])

x, t, n = conjugate_gradient(Q, b, x0) 
print("Convergence: {}\n\nConverged to {} in {} iteration(s)".format(t, x, n))
print("All close to [0.5, 2.0] ? --> {}\n".format(np.allclose(x, np.array([0.5, 2.0])))) 
```

```
Convergence: True

Converged to [0.5 2. ] in 2 iteration(s)
All close to [0.5, 2.0] ? --> True
``` 

Using this method is nifty since at eaach iteration, the routine minimizes the objective function over a subspace that has a dimension equal to the iteration number. This also guaarantees that we will converrge to a minimizer. The method also has convenient features of orthogoality that reduce computational complexity. 


[back](./) 