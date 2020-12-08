import numpy as np
import pandas as pd
import sys

import cost_functions
import cso

import re
import nltk
import math
from nltk.tokenize import word_tokenize
import pandas as pd
import math

#nltk.download('punkt')

def clean_text(file_name):
    file = open(file_name, "r",encoding="utf8")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    # removing special characters and extra whitespaces
    for sentence in article:
        sentence = re.sub('[^a-zA-Z]',' ',str(sentence))
        sentence = re.sub('[\s+]',' ', sentence)
        sentences.append(sentence)
    sentences.pop()
    display = " ".join(sentences)
    print('Initial Text: ')
    print(display)
    print('\n')
    return sentences

def count_words(sent):
    count = 0
    words = word_tokenize(sent)
    for word in words:
        count = count + 1
    return count

# getting data about each sentence (frequency of words)
def count_in_sentence(sentences):
    txt_data = []
    for index,sentence in enumerate(sentences):
        count = count_words(sentence)
        temp = {'id' : index, 'word_cnt' : count}
        txt_data.append(temp)
    return txt_data

def freq_dict(sentences):
    freq_list = []
    for index,sentence in enumerate(sentences):
        freq_dict = {}
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            freq_dict[word] = freq_dict.get(word,0) + 1
        temp = {'id' : index, 'freq_dict' : freq_dict}
        freq_list.append(temp)
    return freq_list

def calc_TF(text_data, freq_list):
    tf_scores = []
    for item in freq_list:
        ID = item['id']
        for key,value in item['freq_dict'].items():
            temp = {
                'id': item['id'],
                'tf_score': value/text_data[ID]['word_cnt'],
                'key': key
            }
            tf_scores.append(temp)
    return tf_scores

def calc_IDF(text_data, freq_list):
    idf_scores = []
    for index,item in enumerate(freq_list):
        for key in item['freq_dict']:
            val = sum([key in it['freq_dict'] for it in freq_list])
            temp = {
                'id':index,
                'idf_score': math.log(len(text_data)/(val+1)),
                'key': key
            }
            idf_scores.append(temp)
    return idf_scores

def calc_TFIDF(tf_scores,idf_scores):
    tfidf_scores = []
    for idf in idf_scores:
        for tf in tf_scores:
            if(idf['key'] == tf['key'] and idf['id'] == tf['id']):
                temp = {
                    'id': idf['id'],
                    'tfidf_score': idf['idf_score']*tf['tf_score'],
                    'key': idf['key']
                }
                tfidf_scores.append(temp)

    return tfidf_scores

def calculateSentSimilarity(sentences,tfidf,n):

  matrix = [[None for col in range(n)] for row in range(n)]

  for index1 in range(n):
    for index2 in range(n):

      s1,s2 = sentences[index1],sentences[index2]
      set1 = set(s1.split(' '))
      set2 = set(s2.split(' '))
      st = set1.union(set2)

      numerator,deno1,deno2 = 0,0,0

      for word in st:
        numerator = numerator + tfidf[index1].get(word,0)*tfidf[index2].get(word,0)
        deno1     = deno1     + tfidf[index1].get(word,0)*tfidf[index1].get(word,0)
        deno2     = deno2     + tfidf[index2].get(word,0)*tfidf[index2].get(word,0)

      score = numerator / (math.sqrt(deno1*deno2))

      matrix[index1][index2] = score

  return matrix

  
def sent_scores(tfidf_scores, sentences, text_data):
    sent_data = []
    for txt in text_data:
        score = 0 
        for index in range(len(tfidf_scores)):
            t_dict = tfidf_scores[index]
            if(txt['id'] == t_dict['id']):
                score = score + t_dict['tfidf_score']
        temp = {
            'id': txt['id'],
            'score': score,
            'sentence': sentences[txt['id']]
        }
        sent_data.append(temp)
    return sent_data


def summary(sent_data):
    count = 0
    summary = []
    for t_dict in sent_data:
        count = count + t_dict['score']
    avg = count / len(sent_data)

    for sent in sent_data:
        if(sent['score'] >= avg*0.9):
            summary.append(sent['sentence'])
    summary = ". ".join(summary)
    return summary

sentences = clean_text('input.txt')
numberSent = len(sentences) 

text_data = count_in_sentence(sentences)

freq_list = freq_dict(sentences)
tf_scores = calc_TF(text_data, freq_list)
idf_scores = calc_IDF(text_data, freq_list)

tfidf_scores = calc_TFIDF(tf_scores, idf_scores)

map = {}
for em in tfidf_scores:
  id,score,word = em['id'],em['tfidf_score'],em['key']
  if(map.get(id,None) == None):
    map[id] = {}
  map[id][word] = score

sentSim = calculateSentSimilarity(sentences,map,numberSent)
sentSim.sort()

# for v in sentSim:
#   print(v)

sent_data = sent_scores(tfidf_scores, sentences, text_data)
result = summary(sent_data)

#print('SUMMARY:')
#print(result)

# tfidf score as position to cso
pos=[]
x=[]
c=0
for t in tfidf_scores:
  if(t['id']!=c):
    pos.append(x)
    x=[]
    c+=1
  x.append(t['tfidf_score'])
pos.append(x)


NUM_RUNS = 1
NUM_CATS = len(sent_data)
MR = 2 #percentage
SMP = 5 #seeking memory pool
SRD = 20 #percentage - seeking range of the selected dimension
c1 = 2
num_dimensions = 2
v_max = 1

def run_experiment(function_name, num_iteration):
	function = getattr(cost_functions, f"{function_name}_fn")

	results = []
	results_pos = []
	avg = 0
	score_cats={}
        
	for _ in range(NUM_RUNS):
		best, best_pos,score_cats = cso.CSO.run(
			num_iteration, 
			function, 
			num_cats=NUM_CATS, 
			MR=MR, 

			num_dimensions=num_dimensions, 
			v_max=v_max,
                        pos=pos,
                        vel=sentSim,
                        
		)
            
		results_pos.append(best_pos)
		results.append(best)
	
	best_all = min(results)
	best_all_pos = results_pos[results.index(best_all)]

	return best_all, best_all_pos, (sum(results) / len(results)),score_cats


def main():
	functions = [
		"spherical",
		#"rastrigin",
		#"griewank",
		#"rosenbrock"
	]

	max_iterations = [50, 100, 500]

	all_results = []

	for function in functions:
		for num_iteration in max_iterations:
			best, best_pos, avg,score_cats = run_experiment(function, num_iteration)
			print(f"Function={function}, Iterations={num_iteration} | best={format(best, '.10f')}, best_pos={best_pos}, avg={format(avg, '.10f')}")
			ind=0

			
			all_results.append([
				function,
				num_iteration,
				best,
				best_pos,
				avg
			])
	print(len(score_cats))
	
	data = np.array(all_results)
	dataset = pd.DataFrame({
		"function": data[:, 0],
		"iterations": data[:, 1],
		"best score": data[:, 2],
		"best position": data[:, 3],
		"avg": data[:, 4]
	})

	dataset.to_excel("results.xlsx")
	print(dataset)

if __name__ == "__main__":
	main()
