---
layout: default
---

## Statistical Arbitrage and Pairs Trading 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 


Pairs trading is a way to take advantage of mean reversion properties in the spread between two assets, or the price difference between them. While we may not know the direction the market is going, pairs trading allows us to place trades based on extremes in the spread between two assets under the assumption that this spread will likely revert to its long-term average. This is done by studying the distribution of this spread, and taking a long position in oone asset and a short position in the other. For this example, we'll look at a pairs trade between the micro Nasdaq (/MNQ) and micro Russell (/M2K) futures in a 2:3 ratio. 

What's nice about pairs trades is that they rely only on the correlation between two assets, not the direction the market is trending in. To do this though, we need to sanity check our trades with some statistics. 

<img src="pairs_trade_exploration.jpg" width="1100" height="500">

<img src="spread_normality.jpg" width="1100" height="425">

<img src="z_scores_moving_averages.jpg" width="1100" height="425">

<img src="final_trade_signals.jpg" width="1100" height="300"> 

