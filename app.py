from flask import Flask,request, url_for, redirect, render_template
import pickle
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
df = pd.read_csv('movie.csv')
model = pickle.load(open('model.pkl','rb'))


cosine_sim = cosine_similarity(model)
def find_title_from_index(index):
    return df[df.Rank == index]["Title"].values[0]
def find_index_from_title(title):
    if len(df[df.Title == title])==0:
        return -1
    return df[df.Title == title]["Rank"].values[0]

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])

def predict():
    movie_name = request.form['query']
    movie_index = find_index_from_title(movie_name)
    if movie_index==-1:
        return render_template('home.html', Your_Movie='{}'.format("Oops No Match Found"))
    similar_movies = list(enumerate(cosine_sim[movie_index]))


    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

    i = 0
    lst=[]
    for element in sorted_similar_movies:
        lst.append(find_title_from_index(element[0]))
        i = i+1
        if i>5:
            break

    return render_template('home.html', Your_Movie='{}'.format(random.choice(lst)))








if __name__ == '__main__':
    app.run(debug=True)