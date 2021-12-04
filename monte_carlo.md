---
layout: default
---

## Integrating Multivariate Functions with Monte Carlo Methods

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>  


## Motivation

Many multivariate functions like the Standard Normal Distribution's pdf cannot be symbolically integrated because their antiderivative does not exist. Quadrature methods are useful in most one-dimensional settings, but do not provide robust integrations in high-dimensions. Monte Carlo sampling provides an efficient (albeit slow) solution for high dimensional integration. 


### The n-Ball 

The open unit n-Ball is a ball that exists in n-dimensional space with radius 1. 


<p><span class="math display">\[U_n = {x\in R^n : ||x||_2 \leq 1>} \]</span></p> 

