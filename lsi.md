---
layout: default
---

## Latent Semantic Indexing Comparing State of the Union Addresses from 1945 to 2013 

<script type="text/javascript" async="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>


```python
import os
import string
import numpy as np
from math import log
from scipy import sparse
from sklearn import datasets
from scipy import linalg as la
from collections import Counter
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla
from sklearn.pipeline import Pipeline
```

Latent Semantic Indexing involves using PCA to reduce the dimensionality of a large corpus of text documents and 
allows us to determine the similarity between two documents' contents. We can represent these documents numerically with a bag of words approach. We first create an ordered set of vocabulary words used in all the documents we are interested in, then cast each document to a numeric vector where the i-th element in each vector is how many times word i appeared in that document. For a set of n documents and vocabulary of m words, this gives us an n x m matrix. 

For a large set of documents and words, this matrix is likely to be both very large and sparse. We can use PCA to reduce the dimension of the data, but we don't scale the matrix in order to retain this sparsity. 

A key feature of this approach to respresenting our documents as a numeric matrix is that we have mapped the text into an inner product space, which allows us to use cosine laws on the vectors in the matrix. For a given document i, we can find the document most similar to it using cosine laws. (It's worth noting that this implicitely is assuming that documents that have similar word frequency distributions are similar to each other and is what we are optimizing for. Other methods build on this )

Thus, finding the document most similar to document i is a matter of finding 

<p><span class="math display">\[argmax_{j\neq i} \frac{< x_i, x_j >}{||x_i|| ||x_j||} \]</span></p> 

This is implemented in the function below. 


```python
def similar(i, Xhat):
    """
    Takes an index and matrix representing the principal components and returns the indices of
    the documents that are the most and least similar to i.
    
    Parameters:
    i index of a document
    Xhat decomposed data
    
    Returns:
    index_max: index of the document most similar to document i
    index_min: index of the document least similar to document i    
    """ 
    # i-th document/row 
    xi = Xhat[i]  
    # find its norm 
    xi_norm = np.linalg.norm(xi, ord = 2)    
    
    # computes cosine similarity between vectors/obs xi and xj 
    helper = lambda xi, xi_norm, xj: np.dot(xi, xj) / (xi_norm * np.linalg.norm(xj, ord = 2)) 
    
    # compute vector similarities and sort indices least --> greatest 
    cosines = np.argsort([helper(xi, xi_norm, Xhat[j]) for j in range(len(Xhat))]) 
    
    return cosines[-2], cosines[0]        
```

Now that we understand how we can mathematically compare written text, let's dig a little deeper into our bag of words approach. Simple bag of words methods can be too simplistic since mere word counts make no comment on how important a word may be in the context of a document (and to every text there is a context!). Most all documents will contain a set of very similar words that are used very often (like, a, the, and, etc) and which contribute little to our overall understanding of the content of the document's semantics. 

More specific words may contain a high information content even if they are not mentioned many times in a document. For example, speaches that mention OPEC and inflation in the 80's are likely to be more related than speeches that just mention "oil." It would make sense that we would need to be able to implement a global weighting over the entire corpus of words to emphasize these feaatures. 

Below I implement a naive bag of words approach to LSI with no global weighting on the words in our corpus, and then I implement a globally weighted version as well as explained below. The function `document_converter` implements the unweighted version by  creating a set of the unique words in all State of the Union Addresses we have, and filters out very common words such as "like," "the," "and," and so on. It also filters punctuation. It returns a matrix X as described above, then I implement PCA through the SVD.   


```python
def unweighted_lsi(speech, l = 7):  
    """
    Uses LSI, applied to the word count matrix X, with the first 7 principal
    components to find the most similar and least similar speeches

    Parameters:
        speech str: Path to speech eg: "./Addresses/1984-Reagan.txt"
        l (int): Number of principal components

    Returns:
        tuple of str: (Most similar speech, least similar speech)
    """ 
    # init X, paths    
    X, paths = document_converter() 

    # get specific speech 
    speech_index = paths.index(speech)  

    # truncated SVD of X 
    U, e, Vt = sparse.linalg.svds(X, k = l)  

    # transform basis of data 
    transformed = sparse.csr_matrix(U).multiply(e).todense()  

    # get indices of most and least similar documents 
    most_similar_index, least_similar_index = similar(speech_index, np.array(transformed))        

    # return paths of those documents 
    return paths[most_similar_index][12: ], paths[least_similar_index][12: ] 
```

```python
speech = "./Addresses/1984-Reagan.txt"
print(unweighted_lsi(speech), "\n") 

speech = "./Addresses/1993-Clinton.txt"
print(unweighted_lsi(speech)) 
```

```
('1988-Reagan.txt', '1946-Truman.txt') 

('2010-Obama.txt', '1951-Truman.txt')
```


We can incorporate a global weighting on our corpus in a few ways, but we'll use the following. 

For each document, i, we'll find the relative word frequency as 

<p><span class="math display">\[p_{i, j} = \frac{X_{i, j}}{\Sigma_{j} X_{i, j}}\]</span></p>  

then we can define the global weight for each word j as 

<p><span class="math display">\[weight_{j} = 1 + \Sigma_{i=1}^{m} \frac{p_{i, j}log(p_{i, j} + 1)}{log(m)} \]</span></p>   

Note how the above formula is reminiscent of the statistical entropy formula, but adjusted for the number of documents in our corpus. With this global weight we replace each element Xij in X with its globally-weighted counterpart

<p><span class="math display">\[ weight_{j} log(X_{i, j} + 1) \]</span></p>    

This is implemented below using a Scipy Sparse matrix. Then we rerun our LSI using our globally-weighted matrix. 


```python 
# init sparse matrix 
x = sparse.csr_matrix((counts, [doc_index, word_index]), shape = (len(paths), len(vocab)), dtype = np.float)  
# init Pij matrix 
pij = sparse.csr_matrix(x / np.sum(x, axis = 0)) 
# get global weights  
gj = (pij.multiply(pij.log1p()).toarray().sum(axis = 0)/np.log(len(vocab))) + 1 
# construct X from weights   
X = x.log1p().multiply(gj) 
``` 

```python
def weighted_lsi(speech, l = 7):  
    """
    Uses LSI, applied to the globally weighted word count matrix A, with the
    first 7 principal components to find the most similar and least similar speeches

    Parameters:
        speech str: Path to speech eg: "./Addresses/1984-Reagan.txt"
        l (int): Number of principal components

    Returns:
        tuple of str: (Most similar speech, least similar speech)
    """
    # init X, paths 
    X, paths = weighted_document_converter() 

    # get specific speech 
    speech_index = paths.index(speech)  

    # do PCA to data to change basis, reduce dimension 
    transformed_a = PCA(n_components = l).fit_transform(X.toarray())  

    # indices of most and least similar rows 
    most_similar_index, least_similar_index = similar(speech_index, transformed_a)         

    # return paths of those documents 
    return paths[most_similar_index][12: ], paths[least_similar_index][12: ]  
```

```python
speech = './Addresses/1984-Reagan.txt'
print(weighted_lsi(speech), "\n")  

speech = "./Addresses/1993-Clinton.txt"
print(weighted_lsi(speech))  
```

```
('1985-Reagan.txt', '1961-Kennedy.txt')

('1994-Clinton.txt', '1951-Truman.txt')
```

One weakness of LSI is that it isnt necessarily robust to adding new documents to the corpus and total vocabulary used. For each new document we wish to include, we'd ideally update our global weights and use the updated document matrix when computing the cosine similarities between documents. Overeall though, LSI is very useful and tons of information retrieval systems rely on similar methods. 

[back](./)