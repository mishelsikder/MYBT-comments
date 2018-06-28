

import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB 

#py.init_notebook_mode(connected = True)

train = pd.read_csv('./mbti_1.csv')
us = pd.read_csv('./Users.csv')
ps = pd.read_csv('./ForumMessages.csv')
mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}

# train.shape

number_types = train['type'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(number_types.index, number_types.values, alpha=0.8)
plt.ylabel('Number of occurenes', fontsize = 12)
plt.xlabel('Types', fontsize=12)
plt.show()

ps['Message'] = ps['Message']ces.fillna('')
ps_join = ps.groupby('AuthorUserId')['Message'].agg(lambda col: ' '.join(col)).reset_index()


