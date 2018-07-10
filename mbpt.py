

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

# train the shape and visualize the dataset

number_types = train['type'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(number_types.index, number_types.values, alpha=0.8)
plt.ylabel('Number of occurenes', fontsize = 12)
plt.xlabel('Types', fontsize=12)
plt.show()

#evaluate performance in the kernel

etc = ExtraTreesClassifier(n_estimators = 20, max_depth=4, n_jobs = -1)
tfidvec = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
trunc_svd = TruncatedSVD(n_components=10)

model = Pipeline([('tfidfvec1', tfidvec), ('trunc_svd1', trunc_svd), ('etc', etc)])

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

scoring = {'acc': 'accuracy',
           #'neg_log_loss': 'neg_log_loss',
           #'f1_micro': 'f1_micro'}


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

 train['clean_posts'] = train['posts'].apply(cleanText)

 #apply Naive Bayes algorithm on the model

 np.random.seed(1)

tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 5000)

model_nb = Pipeline([('tfidf1', tfidf2), ('nb', MultinomialNB())])

results_nb = cross_validate(model_nb, train['clean_posts'], train['type'], cv=kfolds, 
                          scoring=scoring, n_jobs=-1)



print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_acc']),
                                                          np.std(results_nb['test_acc'])))

print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_f1_micro']),
                                                          np.std(results_nb['test_f1_micro'])))

print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_nb['test_neg_log_loss']),
                                                          np.std(-1*results_nb['test_neg_log_loss'])))



#checking the learning curve to see if the model is overwriting

train_sizes, train_scores, test_scores = \
    learning_curve(model_lr, train['clean_posts'], train['type'], cv=kfolds, n_jobs=-1, 
                   scoring="f1_micro", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):

    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    plot_learning_curve(train['posts'], train['type'], train_sizes, 
                    train_scores, test_scores, ylim=(0.1, 1.01), figsize=(14,6))
plt.show()

# test the model on data obtained from comments of users

ps_join['clean_comments'] = ps_join['Message'].apply(cleanText)

model_lr.fit(train['clean_posts'], train['type'])
pred_all = model_lr.predict(ps_join['clean_comments'])

cnt_all = np.unique(pred_all, return_counts=True)

pred_df = pd.DataFrame({'personality': cnt_all[0], 'count': cnt_all[1]},
                      columns=['personality', 'count'], index=None)

pred_df.sort_values('count', ascending=False, inplace=True)

plt.figure(figsize=(12,6))
sns.barplot(pred_df['personality'], pred_df['count'], alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Personality', fontsize=12)
plt.show()


pred_df['percent'] = pred_df['count']/pred_df['count'].sum()

pred_df['description'] = pred_df['personality'].apply(lambda x: ' '.join([mbti[l] for l in list(x)]))

pred_df


labels = pred_df['description']
sizes = pred_df['percent']*100

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Online Personality Distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
