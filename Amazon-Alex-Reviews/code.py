# --------------
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv(path, sep = '\t')
print(type(df['date']))
# Converting date attribute from string to datetime.date datatype 
df['date'] =  pd.to_datetime(df['date'])

# calculate the total length of word
df['length'] = df['verified_reviews'].str.len()



# --------------
## Rating vs feedback

# set figure size
plt.figure(figsize=(15,8))

# generate countplot
sns.countplot(x = 'rating', hue = 'feedback', data = df)
plt.title('Rating V/s Feedback')
plt.xlabel('Rating')
# display plot
plt.show()

## Product rating vs feedback

# set figure size
plt.figure(figsize = (15,8))

# generate barplot
sns.barplot(x = 'rating', y = 'variation', hue = 'feedback', data = df)
plt.title('Variation V/s Feedback')
plt.xlabel('Variation')
# display plot
plt.show()



# --------------
# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# declare empty list 'corpus'
corpus = []

# for loop to fill in corpus
for i in range(0,3150) :
    review = df['verified_reviews'].iloc[i]
    # retain alphabets
    review = re.sub("[^a-zA-Z]", " ", review)
    # convert to lower case
    review = review.lower()
    # tokenize
    review = review.split()
    # initialize stemmer object
    ps = PorterStemmer()
    # perform stemming
    review = [word for word in review if word not in stopwords.words('english')]
    review = [ps.stem(i) for i in review]
    # join elements of list
    review = ' '.join(review)
    # add to 'corpus'
    corpus.append(review)
    
# display 'corpus'
print(corpus)


# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
cv = CountVectorizer(max_features = 1500)

# Independent variable
X = cv.fit_transform(corpus).toarray()

# dependent variable
y = df['feedback']

# Counts
count = df['feedback'].value_counts()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Instantiate calssifier
rf = RandomForestClassifier(random_state = 2)

# fit model on training data
rf.fit(X_train, y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)

# calculate the precision
precision = precision_score(y_test, y_pred)

# display 'score' and 'precision'
print("Accuracy Score", score)

print("Precision Score", precision)


# --------------
# import packages
from imblearn.over_sampling import SMOTE

# Instantiate smote
smote = SMOTE(random_state = 9)

# fit_sample onm training data
X_train, y_train = smote.fit_sample(X_train, y_train)

# fit modelk on training data
rf.fit(X_train, y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)

# calculate the precision
precision = precision_score(y_test, y_pred)

# display 'score' and 'precision'
print("Accuracy Score", score)

print("Precision Score", precision)



