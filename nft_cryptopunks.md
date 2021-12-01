---
layout: default
---

## Generating Original Cryptopunk NFT's with Kernel Density Estimates 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script> 


If you clicked on this, you probably already know what NFTs are. If not, I’ll give a short explanation. 

NFT stands for non-fungible token. Something that is “fungible” is easily exchanged for or comparable to something similar to itself. A one-dollar bill is fungible because it can easily be exchanged for another dollar bill, or even four quarters. Non-fungible basically means “unique.” A “token” in this context is a crypto project built on top of an existing blockchain it is not native to. For example, Ethereum is a blockchain whose native cryptocurrency is called Ether. Any other crypto project built on top of the Ethereum blockchain would be considered a token. NFTs are in large part built on top of the Ethereum blockchain (hence “token”) and are basically like having title to a specific asset in the sense that they ensure ownership of a specific, unique thing. 

NFTs have become very popular recently and the “Cryptopunks” are some of the most famous ones out there. They're very simple pixelated images of people with funny hats, cool hair, and are often smoking. All of them are made in a very similar style, but each is unique. There are about 10,000 original Cryptopunks and they often sell on OpenSea (an NFT marketplace) for big bucks. 

<img src="nft_samples.jpg" width="1000" height="575"> 

A lot of notebooks I’ve seen of people trying to create original Cryptopunks use models like GANs and other deep learning architectures, but given the simplicity of the images, I can’t help but think this might be overkill.  

I decided to try and use Kernel Density Estimators instead. KDEs are some of my favorite models out there, and they walk the line between unsupervised learning, feature engineering, and data modeling. Using them here basically amounts to estimating the pixel distributions of the Cryptopunks, and then sampling new images from that estimated distribution. 

This was very straightforward, and training the KDE took a matter of seconds, rather than minutes or even hours as a fully architectured Neural Network might take. 

```python
%time kde = KernelDensity().fit(all_images)  
```
```
CPU times: user 3.01 s, sys: 105 ms, total: 3.12 s
Wall time: 3.11 s
```

Below are a few examples of original Cryptopunks with my new versions sampled from my KDE. 

<img src="newly_sampled_cryptopunks_trained_on_all_data.jpg" width="1000" height="600"> 


It’s pretty clear that the newly generated KDEs are very similar to the originals in terms of style, but I think this speaks more to the simplicity of the images than the quality of the model. That being said, KDEs are not to be underestimated! 
