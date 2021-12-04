---
layout: default
---

## Integrating Multivariate Functions with Monte Carlo Methods

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>  


```
import numpy as np
import scipy.stats 
import scipy.linalg as la 
import matplotlib.pyplot as plt 
```


## Motivation

Many multivariate functions like the Standard Normal Distribution's pdf cannot be symbolically integrated because their antiderivative does not exist. Quadrature methods are useful in most one-dimensional settings, but do not provide robust integrations in high-dimensions. Monte Carlo sampling provides an efficient (albeit slow) solution for high dimensional integration. 


### The n-Ball 

The open unit n-Ball is a ball that exists in n-dimensional space with radius 1. In one dimension this is a point, in two dimensions this is a circle, a sphere in 3 dimensions, and so on. In terms of set notation, we have 


<p><span class="math display">\[U_n = \{x\in R^n : ||x||_2 < 1\} \]</span></p> 


Luckily, we can easily implement this through code. 


```python 
def ball_volume(n, num_samples = 10000):
    """
    Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
                 n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # uniformly sample a grid of points in n dimensions 
    points = np.random.uniform(-1, 1, (n, num_samples))  

    # take norms from origin (these is how we enforce r < 1)  
    lengths = la.norm(points, axis = 0)

    # return our estimated volume via integration  
    return 2**n * np.count_nonzero(lengths <= 1) / num_samples 
```

```python 

```

```python 

```

```python 

```

```python 

```

```python 

```

```python 

```

```python 

```