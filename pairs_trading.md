---
layout: default
---

## Statistical Arbitrage and Pairs Trading 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

Pairs trading is a market-neutral way to take advantage of mean reversion properties in the spread between two highly correlated assets. This is done by simultaneously taking a long position in one asset and a short position in the other. The strategy relies on arbitraging the distribution of the spread between the two asset prices, or the difference between the prices.  For this example, we’ll look at a pairs trade between the micro Nasdaq (/MNQ) and micro Russell (/M2K) futures in a 2:3 ratio, where the spread is 3/M2K - 2/MNQ. 

While we may not know the direction the market is going, pairs trading allows us to place trades based on extremes in the spread differential between the two assets under the assumption that this spread will likely revert to its long-term average. This makes more sense once we explore the distribution of this spread over time. Let’s start with a quick visual. 

<img src="pairs_trade_exploration.jpg" width="1100" height="500">

In the first visual, we have the cumulative percentage returns for each asset for the week of November 8 through November 12 of 2021. Clearly, both assets closed the week for a loss (this week there was a particularly poor inflation reading of 6.2%). 

The second visual plots the spread between the cumulative percentage returns, or the cumulative percentage returns for 3/M2K contracts minus the cumulative percentage returns for 2/MNQ contracts. This looks like it may be normally distributed, but we’ll explore this further below. 

The third visual plots the P/L (profit vs. losses) for longing 3/M2K contracts and shorting 2/MNQ contracts and holding that trade through the entire week. During a week where both assets closed for losses, the pair trade was largely profitable throughout the entire week. This is still relaativelt inefficient though, since there were clearly more optimal times to enter and exit the trade rather then just holding it for the entire week. 

In order to be confident in our mean reversion strategy, we need to further examine the spread for some statistical sanity checks. Below I plot the spread with its average value over the week, along with a probability plot, where the spread is plotted against a normal distribution to compare percentiles. It appears that within +/- 3 standard deviations it is relatively normally distributed, with skew creeping in for moves greater than 3 standard deviations in magnitude. This gives some credibility to our strategy, since we can use a rolling Z-Score value as a heuristic to generate buy and sell signals. 

<img src="spread_normality.jpg" width="1300" height="450">

To smooth out noise in the spread as we generate buy and sell signals, I’ll use the 5 and 55 moving averages. Ideally, we’d like to enter this trade when it’s relatively cheap and sell when it’s more expensive. To do this, we’ll generate a rolling Z-Score and buy the spread when the Z-Score is negative, and sell the spread when it’s positive, as seen in the visual below. 

The Z-Scores are defined as 

<p><span class="math display">\[ Z = \frac{MA(5) - MA(55)}{\sigma_{MA(55)}} \]</span></p> 

where MA() indicates the moving average and sigma indicates a standard deviation. 

I also test for stationarity of the rolling Z-Scores as well. Since we are using this as our heuristic, stationarity will tell us if the mean, variance, and standard deviation of the rolling Z-Scores change over time. Ideally they would not be changing drastically, since this would allow us to use a simple numeric cutoff to define when we enter or exit the trade. I test this with the augmented Dickey-Fuller test against the null-hypothesis of non-stationarity. Indeed, at a 5% (and even 1%) level, the rolling Z-Scores appear to be stationary. 

```python 
spread_dickey_fuller = adfuller(z_scores.dropna())
print('P-value for the Augmented Dickey-Fuller Test is', spread_dickey_fuller[1])   
```
```
P-value for the Augmented Dickey-Fuller Test is 1.8628096799053177e-05 
```

With this P-value being less than 0.05, we reject the null hypothesis of non-stationarity. 

Now that we have established evidence for stationarity, we’ll define buy and sell signals as follows

<p><span class="math display">\[ buy = Z-score < -1.5, sell = Z-score > 1.5 \]</span></p> 

These cutoffs are plotted in the visual below. 

<img src="z_scores_moving_averages.jpg" width="1100" height="425">

Below is a visual of the spread for 3/M2K - 2/MNQ with our buy and sell signals added in. This strategy does a relatively decent job at buying at lows in the spread aand selling when the spread is higher, though there is lots of noise in the beginning of the week. This strategy can be further improved by modeling the rolling Z-Scores as an autoregressive process, since stationary processes can be well approximatede by auto-regresseve models. Further improvements might include fourier analysis of the spread or implementing a Kalman Filter to update the moving averages in near-real time. Overall, this simplistic approach provides an easily understandable heuristic for entering and exiting trades. 

<img src="final_trade_signals.jpg" width="1100" height="300"> 

(It's worth noting that these signals only indicate whether we buy the spread (long 3/M2K, short 2/MNQ), or sell the spread (short 3/M2K, long 2/MNQ). Determining when to close a position after entering a trade based on these signals is a whole other beast to be tackled in a subsequent project. Further, I'm not a financial advisor, so don't use these metrics for your own investing.) 

