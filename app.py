from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask import Flask
from flask import request
from flask import jsonify
import re
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub

print("new")
model = hub.load(r"https://tfhub.dev/google/universal-sentence-encoder-large/5")
print("model loaded")

app = Flask(__name__)

def preprocess(c):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', c)

@app.route('/duplication', methods=['GET','POST'])
def duplication():
   req = request.get_json()
   fn1 = req['article-content1']
   fn2 = req['article-content2']
   fn1 = preprocess(fn1)
   fn2 = preprocess(fn2)
   base_document = fn1
   documents = [fn2]
       
   base_embeddings = model([base_document])
       
   embeddings = model(documents)
       
   scores = cosine_similarity(base_embeddings, embeddings).flatten()
   highest_score = 0
   highest_score_index = 0
   for i, score in enumerate(scores):
       if highest_score < score:
           highest_score = score
           highest_score_index = i
       
   most_similar_document = documents[highest_score_index]

   sim_score = {'Similarity Score': int(highest_score*100)}
    
   
   y = json.dumps(sim_score)
   return y


if __name__ == '__main__':
   app.run()

   
   
