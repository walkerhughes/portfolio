---
layout: default
---

## Synthetic Control Methods
## Estimating the Effects of COVID-19 'Super Spreader' Events

<p>Check out my full report on this with other identification strategies <a href="./econ488.pdf">here</a></p> 

Control groups are a key part of any scientific study when we want to see the effect of a 'treatment' on a specific group. In medicine, this could take the form of a randomized trial where one half of the study group is given medicine and the other half is given a placebo. We call the medicine group the 'treated group' and the placebo group the control. The control group allows us to see the effect of the treatment in the counterfactual state of never having received the medicine. The key assumption here is that both the treatment and control groups are similar to each other on other observable characteristics and only differ on the basis of receiving the medicine or not. 

In comparative case studies, it can be difficult to find an adequate control group for a given treated unit. In these cases, we can construct a synthetic control group by taking a weighted average of groups that are similar to the treated group, and picking the weights for this weighted average in a way that closely tracks the treated group before treatment began. As an example, we will look at the Sturgis Motorcycle Rally that happened in August 2020 in Meade County, South Dakota. The rally draws thousands of people annually from all overe the US and it pressed on in the face of the COVID pandemic with minimal social distincing enforced. This begs the question: did this rally lead to the increase in COVID-19 cases reported in Meade County shortly after the rally ended? 

Since we will never be able to know what COVID infection numbers would have been in Meade County had the rally not happened, we can create a synthetic control version of Meade County and use that to compare to the true observed cases. We will do this by finding other counties in the US that had similar case numbers prior to the rally and then we'll create a weighted average of them such that the case numbers of the weighted aaverage prior to the rally - August 7, 2020 - are similar to the true case numbers observed in Meade County. We then use this weighted average (our synthetic control for Meade) to forecast hypothetical cases for Meade had the rally not happened, and compare those forecasts to the truth.  

We can sanity check our results by performing placebo tests, or perturbation tests. For each county used in constructing the synthetic control (the donor counties), we create a new synthetic control in the same way out of the other doner counties. We then similarly forecast out in time case number estimates after the onset of the rally and compare our estimates to the true case numbers for that county. These differences should be smaller in magnitude than those between true case numberes in Meade County and its synthetic control. 

This was also fun to implement from scratch in python since decent open-source python packages for synthetic controls seem few and far between. I implemented the optimization routine for finding weights for each synthetic control using the CVXPY library, and included L1 rergularization since the vector of these weights is ideally sparse to avoid overfitting the the pre-rally time period.

Using 21 US counties that best matched Meade County's COVID case numbers prior to August 7, 2020, we get the following rersults. The synthetic control for Meade County clearly shows fewer cases had the Rally not occured, and the difference between true case numbers and the synthetic control case numbers were farr more extreme for Meade County than forr any of the donor counties. This all suggests that the Rally likely had a significant effect on the rise in COVID cases in Meade County. 

The following results suggest that COVID-19 cases in Meade county wouold have been a mere 33% of the true total within two weeks of the rally's end haad the rally now happened. We can sanity check these results in our placebo tests, which suggest that the effect of the rally was the greatest on COVID-19 rates in Meade County relative to our donor pool. 

<p>Check out my synthetic control class on my <a href="https://github.com/walkerhughes/synthetic_control_super_spreader">GitHub</a></p> 

<img src="synth_match_cases.jpg" width="1000" height="1250">    

 