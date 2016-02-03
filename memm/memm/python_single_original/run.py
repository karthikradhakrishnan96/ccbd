from nltk.corpus import treebank as wsj
from memm import MEMM

import cProfile
import pstats
import random
import StringIO

no_of_sentences = 10

def create_dataset():
	print 'Loading dataset'
	dataset = []
	tags = []
	sents = wsj.sents()

	for i,sentence in enumerate(wsj.tagged_sents()[:no_of_sentences]):
		prev = None
		prev_prev = None
		for j,word in enumerate(sentence):
			datapoint = {}
			temp = []
			len_sentence = len(sentence)
			
			if(j > 0):
				temp.append(sents[i][j-1])
			else:
				temp.append('*')
			if(j > 1):
				temp.append(sents[i][j-2])
			else:
				temp.append('*')
			
			temp.append(sents[i][j])

			if(j < len_sentence-1):
				temp.append(sents[i][j+1])
			else:
				temp.append('*')
			if(j < len_sentence-2):
				temp.append(sents[i][j+2])
			else:
				temp.append('*')

			datapoint['wn'] = temp
			
			datapoint['index'] = j
			if(prev == None):
				datapoint['t_minus_one'] = '*'
			else:
				datapoint['t_minus_one'] = prev[1]
			if(prev_prev == None):
				datapoint['t_minus_two'] = '*'
			else:
				datapoint['t_minus_two'] = prev_prev[1]

			prev_prev = prev
			prev = word
			# print datapoint,word[1]
			dataset.append(datapoint)
			tags.append(word[1])
	print 'Done'
	return dataset, tags

def f1(x,y):
	return round(random.random())

def f2(x,y):
	return round(random.random())

def f3(x,y):
	return round(random.random())

def f4(x,y):
	return round(random.random())

def f5(x,y):
	return round(random.random())

def f6(x,y):
	return round(random.random())

def f7(x,y):
	return round(random.random())

def f8(x,y):
	return round(random.random())

def f9(x,y):
	return round(random.random())

def f10(x,y):
	return round(random.random())



if __name__ == '__main__':
	data,tag = create_dataset()
	tag1 = list(set(tag))
	punctuations = ['.', ',', ':',';','\"','\'','``','\'\'']
	all_tag = []
	# for t in tag1:
	# 	if t not in punctuations:
	# 		all_tag.append(t)
	all_tag = list(set(tag))

	param = [0 for i in range(10)]
	print 'Profiling started'
	prof = cProfile.Profile()
	prof.enable()
	memm = MEMM(data, tag, all_tag,param, wsj.tagged_sents()[:no_of_sentences],[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10], 0)
	memm.train()

	prof.disable()
	s = StringIO.StringIO()
	sortby = 'cumulative'
	ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
	ps.print_stats()
	print s.getvalue()
