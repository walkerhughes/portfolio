---
layout: default
---

## Statistical Arbitrage and Pairs Trading 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

Pairs trading is a market-neutral way to take advantage of mean reversion properties in the spread between two highly correlated assets. This is done by simultaneously taking a long position in one asset and a short position in the other. The strategy relies on arbitraging the distribution of the spread between the two asset prices, or the difference between the prices.  For this example, we’ll look at a pairs trade between the micro Nasdaq (/MNQ) and micro Russell (/M2K) futures in a 2:3 ratio, where the spread is 3/M2K - 2/MNQ. 

While we may not know the direction the market is going, pairs trading allows us to place trades based on extremes in the spread between the two assets under the assumption that this spread will likely revert to its long-term average. This makes more sense once we explore the distribution of this spread over time. Let’s start with a quick visual. 

<img src="pairs_trade_exploration.jpg" width="1100" height="500">

In the first visual, we have the cumulative percentage returns for each asset for the week of November 8 through November 12 of 2021. Clearly, both assets closed the week for a loss (this week there was a particularly poor inflation reading of 6.2%). 

The second visual plots the spread between the cumulative percentage returns, or the cumulative percentage returns for 3/M2K contracts minus the cumulative percentage returns for 2/MNQ contracts. This looks like it may be normally distributed, but we’ll explore this further below. 

The third visual plots the P/L (profit vs. losses) for longing 3/M2K contracts and shorting 2/MNQ contracts and holding that trade through the entire week. 

But in order to be confident in our mean reversion strategy, we need to further examine the spread for some statistical sanity checks. Below I plot the spread with its average value over the week, along with a probability plot, where the spread is plotted against a normal distribution to compare percentiles. It appears that within +/- 3 standard deviations it is relatively normally distributed, with skew creeping in for moves greater than 3 standard deviations in magnitude. This gives some credibility to our strategy, since we can use a rolling Z-Score value as a heuristic to generate buy and sell signals. 

<img src="spread_normality.jpg" width="1300" height="450">

To smooth out noise in the spread as we generate buy and sell signals, I’ll use the 5 and 55 moving averages. Ideally, we’d like to enter this trade when it’s relatively cheap and sell when it’s more expensive. To do this, we’ll generate a rolling Z-Score and buy the spread when the Z-Score is negative, and sell the spread when it’s positive, as seen in the visual below. I also test for stationarity of the rolling Z-Scores as well.  

To smooth out noise in the spread as we generate buy and sell signals, I’ll use the 5 and 55 moving averages. Ideally, we’d like to enter this trade when it’s relatively cheap and sell when it’s more expensive. To do this, we’ll generate a rolling Z-Score and buy the spread when the Z-Score is negative, and sell the spread when it’s positive, as seen in the visual below. 

I also test for stationarity of the rolling Z-Scores as well. Since we are using this as our heuristic, stationarity will tell us if the mean, variance, and standard deviation of the rolling Z-Scores change over time. Ideally they would not be changing drastically, since this would allow us to use a simple numeric cutoff to define when we enter or exit the trade. I test this with the augmented Dickey-Fuller test against the null-hypothesis of stationarity. Indeed, at a 5% (and even 1%) level, the rolling Z-Scores appear to be stationary. 

```python 
spread_dickey_fuller = adfuller(z_scores.dropna())
print('P value for the Augmented Dickey-Fuller Test is', spread_dickey_fuller[1])   
```

<img src="z_scores_moving_averages.jpg" width="1100" height="425">

Now that we’ve established stationarity, we’ll define buy and sell signals as follows


<img src="final_trade_signals.jpg" width="1100" height="300"> 

