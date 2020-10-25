# **Recognizing Duplicate Questions using Trax**

**Introduction and Objective**
- I have worked with **Quora Questions Answers Dataset** to build a Long Short Term Memory or LSTM Model that can identify the Similar Questions or the Duplicate Questions which is useful when we have to work with several versions of the same Questions. I have build a Model using Trax which can identify the Duplicate Questions in this Project.

**Neural Networks and Trax**
- Trax is good for Implementing new state of the Art Algorithms like Transformers, Reformers and BERT. It is actively maintained by Google Brain Team for Advanced Deep Learning tasks.

**Siamese Neural Networks**
- Siamese Neural Network is an Artificial Neural Network which uses the same weight while working in tandem on two different input vectors to compute comparable output vectors.  In Siamese Neural Networks: One of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared.

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import pandas as pd
import numpy as np
import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import random
from collections import defaultdict
from functools import partial
```

**Getting the Data**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I will be using Quora Answer Question Dataset for this Project. I will build a Model that can Identify the Similar Questions or the Duplicate Questions which is useful when we have to work with several versions of the same Questions. The Dataset is labeled.

**Processing the Data**
- I will split the Data into Training set and Testing Set. The Test Set will be used later to evaluate the Model. I will select only the Question Pairs that are duplicate to train the Model. I will build two batches as input for the Neural Networks: Siamese Networks. The Test set uses the original pairs of Questions and the Status describing if the Questions are duplicates. I will encode each word of the selected pairs with an Index which will be a list of numbers. Firstly, I will Tokenize each word using NLTK and I will use Python's Default Dictionary which assigns the values 0 to all Out of Vocabulary Words. I have presented the simple Implementation of Data Preparation for training the LSTM Model using Quora Dataset here in the Snapshot.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2049.PNG)

**Siamese Neural Networks and Triplet Loss**
- Siamese Neural Network is an Artificial Neural Network which uses the same weight while working in tandem on two different input vectors to compute comparable output vectors.  In Siamese Neural Networks: One of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared. The Triplet Loss makes use of a Baseline or Anchor Input which is compared to the Positive or Truthy Input and a Negatve or Falsy Input. The distance from the Anchor Input to the Positive Input is minimized and the distance from the Anchor Input to the Negative Input is maximized. The Triplet Loss is composed of two terms where one term utilizes the mean of all the non duplicates and the second term utilizes the Closest Negative. I have presented the Implementation of Siamese Neural Network using LSTM Model along with the Implementation of Triplet Loss here in the Snapshots.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2050a.PNG)

**Training the Model**
- Now, I will train the Model. I will define the Cost Function and the Optimizer as ususal. I will use Training Iterator to go through all the Data for each Epochs while training the Model.  I have also presented the Implementation of Training the Model using Data Generators and other dependencies. I have presented all the Implementations using Trax Framework here in the Snapshots.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2050b.PNG)

**Model Evaluation**
- I will utilize the Test Set which was configured earlier to determine the accuracy of the Model. Actually the Training Set only had Positive examples whereas the Test Set and y test is setup as pairs of Questions and some of which are duplicates and some are not. I will compute the Cosine Similarity of each pair, threshold it and compare the result to y test. The results are accumulated to produce the Accuracy. I have presented the Implementation of Siamese Neural Network using LSTM Model that can identify the Similar or Duplicate Questions here in the Snapshots.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2051.PNG)

**Predicting the Questions Pairs**
- Now, I will test the Model using my own Questions. I will build a reverse Vocabulary that allows the map encoded Questions back to words. I have also presented the output of the Model here which fascinates me a lot. 

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2051b.PNG)
