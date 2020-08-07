import pandas as pd
import numpy as np
from flask import Flask,render_template,redirect,request
import pickle
import requests
from bs4 import BeautifulSoup as bf
from fuzzywuzzy import process
import gc


model = pickle.load(open('knn_model','rb'))
pvt = pickle.load(open('pvt_new','rb'))
movie_names = pickle.load(open('movie_names_list','rb'))

search_url = 'https://www.imdb.com/find?q='
main_url = 'https://www.imdb.com'
hd = 'UX350_CR1,0,360,530_AL__QL100.jpg'

def get_info(movie_name):
    try:
        movie_name = movie_name.split("(")[0]+"("+movie_name.split("(")[-1]
        res = requests.get(search_url+movie_name).text
        soup = bf(res,'html5lib').find(class_='findList').find(class_='findResult odd')
        movie_link = main_url + soup.find(name='a').attrs['href']
        try:
            img_link = soup.find(name='img').attrs['src'][:-24]+hd
        except:
            img_link = 'NaN'
        return movie_link,img_link
    except:
        movie_link = 'NaN'
        img_link = 'NaN'
        return movie_link,img_link


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def check():
    i = request.form.get('movie_name')
    found = [i[0] for i in process.extract(i,movie_names,limit=3) if i[1]>=90]
    if len(found)==1:
        return predict(found[0])
    else:
        return render_template('search.html',found=found,count=len(found))


@app.route('/predict', methods=['POST'])
def predict(movie_name='NaN'):
    if movie_name == 'NaN':
        movie_name = request.form.get('movie_name')
    ID = np.where(pvt.index==movie_name)[0][0]
    sug_id = model.kneighbors(X=pvt.iloc[ID,:].values.reshape(1,-1),n_neighbors=7, return_distance=False)[0]
    data = []
    for i in sug_id:
        x,y = get_info(pvt.index[i])
        data.append((pvt.index[i],x,y))
    del(sug_id)
    
    return render_template('predict.html',data=data) 
gc.collect()


if __name__=='__main__':
    app.run(debug=True)

"""
i = request.form.get('movie_name')
    p = re.compile(f"{i.lower()}?")
    found = []
    for x in movie_names:
        if re.search(p,x.lower()):
            found.append(x)
    found = list(np.unique(np.array(found)))
    l = list(range(len(found)))
    count = len(found)
"""