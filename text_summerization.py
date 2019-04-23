from flask import Flask, flash, redirect, render_template, request
from random import randint
import bs4 as bs
from urllib import urlopen
import re
import nltk
import heapq
import requests
import pandas as pd
import numpy as np
import csv

app = Flask(__name__)




@app.route("/search")
def show():
	# url="https://en.wikipedia.org/wiki/Goat"
	url=request.form['url']

	#r = requests.get(url)
	text1=summ(url)
	text2=summarize(str(text1),word_count=300)

	#return render_template('index.html', **locals())
	return url


@app.route("/", methods = ['POST', 'GET'])
def sho():
	errors = []
    	results = {}
	urltext="https://en.wikipedia.org/wiki/Automatic_summarization"
	senlen = 30
    	if request.method == "POST":
		urltext = request.form['url']
		senlen = request.form['senlen']
		print senlen





	page=urlopen(urltext)
	soup=bs.BeautifulSoup(page,'html.parser')
	images = soup.findAll('img')
	head = soup.findAll('h1')
	heading = head[0].text
	imgsrc = images[1]['src']
	text2=letmeexplain(urltext, senlen)



	return render_template('index.html', **locals())


def letmeexplain(urltext, senlen):
	scraped_data = urlopen(urltext)
	article = scraped_data.read()

	parsed_article = bs.BeautifulSoup(article,'lxml')

	paragraphs = parsed_article.find_all('p')

	article_text = ""

	for p in paragraphs:
    		article_text += p.text

	article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
	article_text = re.sub(r'\s+', ' ', article_text)



	formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
	formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

	sentence_list = nltk.sent_tokenize(article_text)
	#sentence_list = article_text.split('.')


	#stopwords = nltk.corpus.stopwords.words('english')
	stopwords = [
	    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
	    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
	    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
	    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
	    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
	    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
	    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
	    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
	    'with', 'about', 'against', 'between', 'into', 'through', 'during',
	    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
	    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
	    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
	    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
	    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
	    't', 'can', 'will', 'just', 'don', 'should', 'now', 'id', 'var',
	    'function', 'js', 'd', 'script', '\'script', 'fjs', 'document', 'r',
	    'b', 'g', 'e', '\'s', 'c', 'f', 'h', 'l', 'k'
	]

	word_frequencies = {}
	data=formatted_article_text.split()
	for word in data:
	    if word not in stopwords:
        	if word not in word_frequencies.keys():
        	    word_frequencies[word] = 1
        	else:
        	    word_frequencies[word] += 1



	print word_frequencies
	with open('sim_mat.csv', 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(word_frequencies)
	maximum_frequncy = max(word_frequencies.values())
	word_embeddings = {}
	f = open('glove.6B.100d.txt')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    word_embeddings[word] = coefs
	f.close()

	sentence_vectors = []

	for i in sentence_list:
	  if len(i) != 0:
	    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
	  else:
	    v = np.zeros((100,))
	  sentence_vectors.append(v)
	sim_mat = np.zeros([len(sentence_list), len(sentence_list)])

	from sklearn.metrics.pairwise import cosine_similarity
	for i in range(len(sentence_list)):
	  for j in range(len(sentence_list)):
	    if i != j:
	      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


	for word in word_frequencies.keys():
	    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


	print sim_mat
	with open('sen_vec.csv', 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(sentence_vectors)

	with open('sim_mat.csv', 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(sim_mat)

	import matplotlib.pyplot as plt
	import networkx as nx
	nx_graph = nx.from_numpy_array(sim_mat)
	#print nx_graph
	y_pos = np.arange(len(nx_graph))
	plt.bar(y_pos, nx_graph)
	plt.xticks(y_pos, nx_graph)
	plt.show()

	sentence_scores = {}
	for sent in sentence_list:
	    for word in nltk.word_tokenize(sent.lower()):
	        if word in word_frequencies.keys():
	            if len(sent.split(' ')) < 30:
	                if sent not in sentence_scores.keys():
	                    sentence_scores[sent] = word_frequencies[word]
	                else:
	                    sentence_scores[sent] += word_frequencies[word]



	summary_sentences = heapq.nlargest(int(senlen), sentence_scores, key=sentence_scores.get)

	summary = ' '.join(summary_sentences)
	return summary



def summery(url,sen_num):
	scraped_data = urlopen(url)
	article = scraped_data.read()
	parsed_article = bs.BeautifulSoup(article,'lxml')
	paragraphs = parsed_article.find_all('p')
	article_text = ""
	sentence=[]
	for p in paragraphs:
    		article_text += p.text
		sentence.append(nltk.sent_tokenize(p.text))
	pd.Series(sentence).str.replace("[^a-zA-Z]", " ")
	article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
	article_text = re.sub(r'\s+', ' ', article_text)
	formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
	formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
	clean_sentences = nltk.sent_tokenize(article_text)
	word_embeddings = {}
	f = open('glove.6B.100d.txt')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    word_embeddings[word] = coefs
	f.close()
	#print word_embeddings
	clean_sentences = [s.lower() for s in clean_sentences]
	from nltk.corpus import stopwords
	stop_words = stopwords.words('english')
	# function to remove stopwords
	def remove_stopwords(sen):
	    sen_new = " ".join([i for i in sen if i not in stop_words])
	    return sen_new
	# remove stopwords from the sentences
	clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
	print clean_sentences
	# Extract word vectors
	word_embeddings = {}
	f = open('glove.6B.100d.txt')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    word_embeddings[word] = coefs
	f.close()
	sentence_vectors = []
	for i in clean_sentences:
	  if len(i) != 0:
	    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
	  else:
	    v = np.zeros((100,))
	  sentence_vectors.append(v)
	sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])
	print sentence_vectors
	from sklearn.metrics.pairwise import cosine_similarity
	for i in range(len(clean_sentences)):
	  for j in range(len(clean_sentences)):
	    if i != j:
	      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

	with open('sen_vec.csv', 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(sentence_vectors)

	with open('sim_mat.csv', 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(sim_mat)

	import matplotlib.pyplot as plt


	#print sim_mat
	import networkx as nx
	nx_graph = nx.from_numpy_array(sim_mat)
	#print nx_graph
	y_pos = np.arange(len(nx_graph))
	plt.bar(y_pos, nx_graph)
	plt.xticks(y_pos, nx_graph)
	plt.show()
	scores = nx.pagerank(nx_graph)
	ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(clean_sentences)), reverse=True)
	summary=""
	for i in range(int(sen_num)):
  		summary+=ranked_sentences[i][1]
	return summary






if __name__ == "__main__":
	app.run(host='127.0.0.1', port=12310, debug=True)
