---
layout: default
---

## Speech Recognition

<p>Check it out on my <a href="https://github.com/walkerhughes/speech_recognition_cdhmm">GitHub</a></p>

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 

```python
import scipy as sp
import scipy.io.wavfile as wavfile
import os
import gmmhmm as hmm
import MFCC
import numpy as np
import re
import random
import pickle
from tqdm import tqdm 
```

Speech recognition is a cool application of Hidden Markov Models when we allow the state space to be continuous rather than discrete - Continuous Density Hidden Markov Models. Here I use a simple implementation of one type of these, the Gaussian Mixture Model Hidden Markov Model, to classify audio clips of 5 different words.  

A mixture of Gaussians is a distribution composed as a linear combination of M Gaussian (or Normal) distributions. Here we will try to classify sound bits of 5 different words being said, so we will take M = 5. Our model then becomes: 

<p><span class="math display">\[f(x) = \sum_{i = 1} ^{M} c_i N(x; \mu_i, \Sigma_i)\]</span></p> 

In a GMMHMM, we seek to model a sequence of hidden (unobserved) states {x1, . . . , xT} and corresponding
sequence of observations {O1, . . . , OT} where each observation Oi is a vector of length K distributed according to a mixture of Gaussians with M components. The parameters for this type of model include the initial state distribution
π and the state transition matrix A. Also, for each state i, we have an assciated set of parameters 

<p><span class="math display">\[i = {1, ..., M}, (c_i, \mu_i, \Sigma_i)\]</span></p> 

The following function accepts a GMMHMM as well as an integer n_sim, and which simulates the GMMHMM process, generating n_sim different observations. 


```python 

def sample_gmmhmm(gmmhmm, n_sim):
    """
    Simulate from a GMMHMM.
    
    Returns
    -------
    states : ndarray of shape (n_sim,)
        The sequence of states
    obs : ndarray of shape (n_sim, K)
        The generated observations (vectors of length K)
    """
    A, weights, means, covars, pi = gmmhmm 
    states, obs = np.zeros(n_sim), np.zeros((n_sim, len(weights[0]))) 
         
    for i in range(n_sim): 
        # choose initial state
        state = np.argmax(np.random.multinomial(1, pi))
        # randomly sample
        sample_component = np.argmax(np.random.multinomial(1, weights[state,:])) 
        sample = np.random.multivariate_normal(means[state, sample_component, :], 
                                              covars[state, sample_component, :, :])
        # update states and obs arrays   
        states[i], obs[i] = state, sample                                
                                               
    return states, obs

```

Most speech recognition models don't actually recognize words, rather they recognize distinct sounds produced by a language called phonemes. English had 44 unique phonemes, so our implementation will have 44 distinct GMMHMMs, one for each distinct sound, and thus each word can be represented as a combination of some subset fo these 44 sounds. 

Before we can do this, audio data takes a good amount of pre-processing. We will do this be representing each audio clip by its mel-frequency cepstral coefficients (MCFFs). we can train a GMMHMM on various audio clips or 
MFCCs for a given word, and by doing this for several words, we form a collection
of GMMHMMs, one for each word. For a new speech signal, after decomposing it
into its MFCC array, we can score the signal against each GMMHMM, returning the word whose
GMMHMM scored the highest. Further info on this can be found in my GitHib repository for this project. 

```python 
# skip the repeats, keep the mels in mels dict 
repeats = {"Biology00.wav", "Mathematics00.wav", "PoliticalScience.wav", 
           "Psychology00.wav", "Statistics00.wav"}
mels = {'Biology': [], 'Mathematics': [], 'PoliticalScience': [], 
        'Psychology': [], 'Statistics': []} 
 
filepath = "./Samples"

# loop over files 
for doc in os.listdir(filepath): 
    if doc not in repeats:
        temp = doc.split(" ")
        try: 
            # get the mel., append to appropriate list 
            num, x = wavfile.read(filepath + "/" + doc)  
            mel = MFCC.extract(x, show = False)   
            mels[temp[0]].append(mel)  
        except: 
            continue

# unpack lists and make sure there are 30 arrays 
bio, math, polysci, psych, stats = mels.values() 
for l in [bio, math, polysci, psych, stats]: 
    print(len(l), end = " ")

```

Now let's actually train the model! 

```python
words = mels.keys() 
samples = [bio, math, polysci, psych, stats] 

# loop over each word and its samples 
for word, word_samples in zip(words, samples): 
    
    # get train and test data 
    x_train, x_test = word_samples[: 20], word_samples[20: ] 
    best = -np.inf  
    loop = tqdm(range(10)) 
    
    for i in loop: 
        
        # train each model 10 times 
        startprob, transmat = initialize(5)
        model = hmm.GMMHMM(n_components=5, n_mix=3, transmat=transmat, startprob=startprob, cvtype='diag')
        model.covars_prior = 0.01
        model.fit(x_train, init_params='mc', var=0.1)
        
        # track the best model for each word 
        if model.logprob > best: 
            
            best = model.logprob 
            best_model = model 
            
    # save the models 
    pickle.dump(best_model, open("{}.p".format(word), "wb"))

print(best)
```
```
-30489.250566198258
```

```python
# load in the models 
models = [pickle.load(open("{}.p".format(word), "rb")) for word in words] 

accs = {} 

# loop over the words and their samples 
for index, (word, sample) in enumerate(zip(words, samples)): 
    
    # get the test data and label 
    x_test, y_test, preds = sample[20: ], index, [] 

    # get predicted label for each word 
    for obs in x_test: 
        y_hat = np.argmax([model.score(obs) for model in models]) 
        preds.append(y_hat) 
       
    # print results and update accuracy dictionary 
    acc = 100 * np.mean([pred == y_test for pred in preds]) 
    accs.update({word: acc})
    if word == "Biology": 
        print("Accuracy for {}: \t\t{:.2f}%".format(word, acc)) 
    else: 
        print("Accuracy for {}: \t{:.2f}%".format(word, acc))
```

```
Accuracy for Biology: 		100.00%
Accuracy for Mathematics: 	100.00%
Accuracy for PoliticalScience: 	 90.00%
Accuracy for Psychology: 	100.00%
Accuracy for Statistics: 	100.00%
```

[back](./)
