import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import warnings
import pickle
warnings.filterwarnings('ignore')
df = pd.read_csv('movie.csv')

columns = ['Actors','Director','Genre','Title','Rating']


def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(
            data['Actors'][i] + ' ' + data['Director'][i] + ' ' + data['Genre'][i] + ' ' + data['Title'][i] + ' ' + str(
                data['Rating'][i]))
    return important_features

df['important_features'] = get_important_features(df);

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["important_features"])

pickle.dump(count_matrix,open('model.pkl','wb'))


