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

A simple example oof how this is useful is when estimating pi. Pi is an irrational number, so its decimal expansion continues infinitely. We can still estimate it though, oone way being through Monte Carlo integration. We know the area of a circle is A = pi*r^2, where r is the circle radius. If r = 1 then, we have that the area is simply equal to pi itself. Intuitively, if we sample uniformly from a unit square centered at (0, 0), we can estimate pi as the ratio of sampled points from the unit square that fall within a circle with radius 1 inscribed in the square. Adjusting for the measure of the set we sampled from gives us our estimate. 

This explanation is taken from a text from the BYU ACME curriculum. 

<img src="integration.jpg" width="900" height="550"> 



### The n-Ball 

The open unit n-Ball is a ball that exists in n-dimensional space with radius 1. In one dimension this is a point, in two dimensions this is a circle, a sphere in 3 dimensions, and so on. In terms of set notation, we have 

<p><span class="math display">\[U_n = \{x\in R^n : ||x||_2 < 1\} \]</span></p> 

We can easily find the volume of any open n-ball with Monte Carlo integration. The basic idea is to sample uniformly over the domain for integration, apply our function restrictions to those sampled points, and find the portion of points sampled in that domain that meet the restriction that r<1, or that the sampled points have a 2-norm from the origin strictly less than 1. Lucky for us, we know that the volume of [-1, 1] x [-1, 1] x ... x [-1, 1] n times is 2^n. I implement this below. 

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

    # take norms from origin (this is how we enforce r < 1)  
    lengths = la.norm(points, axis = 0)

    # return our estimated volume via integration  
    return 2**n * np.count_nonzero(lengths < 1) / num_samples 
```
```python 
print("Estimated volume of unit sphere: {:.3f}".format(ball_volume(n = 3, N = 10**6)))

print("True value of unit sphere:       {:.3f}".format((4 / 3) * np.pi))

print("Estimated volume of a unit 4-ball is: {}".format(ball_volume(n = 4, N = 10**5))) 
```
```
Estimated volume of unit sphere:      4.183
True value of unit sphere:            4.189
Estimated volume of a unit 4-ball is: 4.91504
```

#### Quick Note on Errors and Convergence 
This performed fairly well for an open unit n-Ball, but our estimates improve as we sample more points to use. Sampling more points greatly increases the temporal complexity of the routine, however. The error of this method is actually proportional to N^(-1/2), where N
is the number of points we sample. Thus, dividing the error by 10 requires 100 times more sample points. When precision is highly important, this may be temporally prohibitive (or would require some parallelization). However, the convergence rate for Monte Carlo integration actually independent of the number of dimensions we're integrating over. Thus, the error for our estimations converges at the same rate when integrating a 2-dimensional function or a 100-dimensional function. This makes Monte Carlo integration preferable over many other machine-integration methods. 

That said, let's apply this to a more practical example.

### Integrating the Standard Normal Probability Density Function 

The Normal Distribution appears all over the place in applied mathematics, statistics, machine learning, and virtually all other quantitative fields. Yet, it has no closed-form solution for its Cumulative Distribution Function, which is the integral of its PDF. Luckily, we can use Monte Carlo for this integration. 

The pdf for an n-dimensional joint Standard Normal distribution (mean of 0 and standard deviation of 1) is as follows 

<p><span class="math display">\[ f(x) = \frac{1}{{2\pi}^{\frac{n}{2}}} e ^ {\frac{-x^{T}x}{2}}  \]</span></p> 


I'll integrate this in 4-dimensional space on the domain [-1.5, 0.75]x[0, 1]x[0, 0.5]x[0, 1] and compare my estimates with the "true" estimates from the `scipy.stats.mvn.mvnun()` method to demonstrate how the error proportional to N^(-1/2). 

```python 
def mc_integrate(f, mins, maxs, num_samples = 10000):
    """
    Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. 
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.
    Returns:
        (float): An approximation of the integral of f over the domain.
    """
    # take the measure of the set we integrate over
    measure = np.product([maxs[i] - mins[i] for i in range(len(mins))])
    
    # saample a grid of points  
    points = np.array([np.random.uniform(a, b, (1, num_samples)) for a, b in zip(mins, maxs)])[:, 0]
    
    # sum the points together 
    f_sum = np.sum([f(points[:, i]) for i in range(len(points[0]))])
    
    # multiply by measure and divide by num_samples 
    return measure * f_sum / num_samples  
```

```python 

```

```python 

```

```python 

```

```python 

```