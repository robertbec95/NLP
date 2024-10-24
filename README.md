1. Open the Ipynb file with Colab
2. Upload the .csv file “dataset.csv”
3. Install Dependencies
Python library that provides a higher-level wrapper that facilitates the use of Deep Learning based NLP models.
• !pip install ktrain -q à ktrain:
Installs a specific version of the eli5 library from the GitHub repository using pip. This version (tfkeras_0_10_1)
is compatible with TensorFlow Keras. The -q option is for quiet mode.
• !pip3 install -q git+https://github.com/amaiya/eli5@tfkeras_0_10_1
Installs the latest version of the eli5-tf library directly from the GitHub repository in zip format. The -q option is
for quiet mode. This library provides tools for interpreting machine learning models trained with TensorFlow.
• !pip3 install -q https://github.com/amaiya/eli5 tf/archive/refs/heads/master.zip
4. Import Libraries and Methods:
from IPython.core.display import display, HTML import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns sns.set_palette(sns.color_palette("seismic")) import sys
import pandas as pd
import numpy as np
import operator
import string
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
