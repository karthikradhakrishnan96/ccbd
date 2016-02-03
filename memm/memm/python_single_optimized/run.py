from nltk.corpus import treebank as wsj
from memm import MEMM

import cProfile
import pstats
import random
import StringIO

import datetime

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
	# check if first letter capital and tag is proper noun
	
	if(x['wn'][2][0].isupper() and (y=='NNP' or y=='NNPS')):
		return 1
	else:
		return 0

def f2(x,y):
	# check if first letter capital and previous tag is NNP
	if(x['wn'][2][0].isupper() and x['t_minus_one'] == 'NNP'):
		return 1
	else:
		return 0	

def f3(x,y):
	# previos tag is adjective and current tag is noun
	if(x['t_minus_one'] in ['JJ','JJR','JJS'] and y in ['NN','NNP','NNPS','NNS']):
		return 1
	else:
		return 0

def f4(x,y):
	# previos tag is adverb and current tag is verb
	if(x['t_minus_one'] in ['RB','RBR','RBS'] and y in ['VBD','VBG','VBN','VBP','VBZ']):
		return 1
	else:
		return 0

def f5(x,y):
	prepositions = ['about','above','across','after','against','along','among','around','at','before','behind','below','beneath','beside','between','by','down','during','except','for','from','in','inside','instead of','into','like','near','of','off','on','onto','out of','outside','over','past','since','through','to','toward','under','underneath','until','up','upon','with','within','without']
	if x['wn'][2].lower() in prepositions and y=='IN':
		return 1
	else:
		return 0

def f6(x,y):
	# wh questions
	if(x['wn'][2][:2].lower() == 'wh' and y in ['WDT','WP','WRB']):
		return 1
	else:
		return 0

def f7(x,y):
	conjunctions = ['and','or','but','nor','so','for', 'yet']
	if x['wn'][2].lower() in conjunctions and y=='CC':
		return 1
	else:
		return 0

def f8(x,y):
	# check if no
	s = x['wn'][2].lower()
	try:
		float(s)
		if(y == 'CD'):
			return 1
		else:
			return 0
	except ValueError:
		return 0

def f9(x,y):
	# check for determiners
	determiners = ['a','an','the','this','that','these','those','my','your','her','his','its','our','their']
	if x['wn'][2].lower() in determiners and y=='DT':
		return 1
	else:
		return 0

def f10(x,y):
	# check for existential 'there'
	# current word is 'there' and next word is the one in list
	if x['wn'][2].lower() =='there' and x['wn'][3].lower() in ['is','was','were','has'] and y=='EX':
		return 1
	else:
		return 0

def f11(x,y):
	# personal pronoun
	pp = ['i', 'you', 'he', 'she', 'it', 'they', 'we']
	if x['wn'][2].lower() in pp and y=='PRP':
		return 1
	else:
		return 0

def f12(x,y):
	if x['wn'][2].lower() == 'to' and y=='TO':
		return 1
	else:
		return 0

def f13(x,y):
	# check if ascii, FW means foreign word
	s = x['wn'][2]
	if(not all(ord(c) < 128 for c in s) and y=='FW'):
		return 1
	else:
		return 0

def f14(x,y):
	# modal verb
	modal = ['can', 'could', 'may', 'might', 'must', 'ought', 'shall', 'should', 'will', 'would']
	if x['wn'][2].lower() in modal and y=='MD':
		return 1
	else:
		return 0

		


# def f1(x,y):
# 	return round(random.random())

# def f2(x,y):
# 	return round(random.random())

# def f3(x,y):
# 	return round(random.random())

# def f4(x,y):
# 	return round(random.random())

# def f5(x,y):
# 	return round(random.random())

# def f6(x,y):
# 	return round(random.random())

# def f7(x,y):
# 	return round(random.random())

# def f8(x,y):
# 	return round(random.random())

# def f9(x,y):
# 	return round(random.random())

# def f10(x,y):
# 	return round(random.random())



if __name__ == '__main__':
	data,tag = create_dataset()
	tag1 = list(set(tag))
	punctuations = ['.', ',', ':',';','\"','\'','``','\'\'']
	all_tag = []
	# for t in tag1:
	# 	if t not in punctuations:
	# 		all_tag.append(t)
	all_tag = list(set(tag))

	param = [0 for i in range(24)]
	# print 'Profiling started'
	# prof = cProfile.Profile()
	# prof.enable()

	memm = MEMM(data, tag, all_tag,param, wsj.tagged_sents()[:no_of_sentences],[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10, f11, f12, f13,f14], 0)
	memm.train()
	print 'For',memm.num_calls_cost,'calls to cost total time taken is',memm.tot_time_cost
	print 'Per call avg time taken is',memm.tot_time_cost/memm.num_calls_cost
	print 'For',memm.num_calls_gradient,'calls to gradient total time taken is',memm.tot_time_gradient
	print 'Per call avg time taken is',memm.tot_time_gradient/memm.num_calls_gradient
	
	# dt1 = datetime.datetime.now()
	# print 'before training: ', dt1
	
	# memm.cost(param)
	# dt2 = datetime.datetime.now()
	# print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()


	# prof.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()
