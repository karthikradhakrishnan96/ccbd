import json
from pyspark import SparkContext, SparkConf
from nltk.corpus import treebank as wsj
import numpy
from pyspark.sql import SQLContext
import datetime
from scipy.optimize import minimize as mymin 
import math
from time import time
import sys
# import pdb
################################################
no_of_sentences = 10
features = []
param = []
all_input_features = []
appName = 'memm'
master = 'spark://10.1.10.92:7077'
master = 'local[3]'



conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext()
sqlContext = SQLContext(sc)


num_calls_cost = 0
num_calls_gradient = 0
tot_time_cost = 0
tot_time_gradient = 0



def create_input_dataset():
	print 'Loading input'
	input_data = []
	tags = []
	sents = wsj.sents()
	json_file  = open('data.json','w') 
	counter = 0
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

			datapoint['i'] = counter
			counter += 1
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
			datapoint['tag'] = word[1]
			json_file.write(json.dumps(datapoint))
			json_file.write('\n')
			input_data.append(datapoint)
			tags.append(word[1])
	print 'Done'
	json_file.close()
	return input_data, tags

def f1(x,y):
	# check if first letter capital and tag is proper noun
	# print x,y

	if(x[5][2][0].isupper() and (y=='NNP' or y=='NNPS')):
		return 1
	else:
		return 0


def f2(x,y):
	# check if first letter capital and previous tag is NNP
	if(x[5][2][0].isupper() and x[2] == 'NNP'):
		return 1
	else:
		return 0	

def f3(x,y):
	# previos tag is adjective and current tag is noun
	if(x[2] in ['JJ','JJR','JJS'] and y in ['NN','NNP','NNPS','NNS']):
		return 1
	else:
		return 0

def f4(x,y):
	# previos tag is adverb and current tag is verb
	if(x[2] in ['RB','RBR','RBS'] and y in ['VBD','VBG','VBN','VBP','VBZ']):
		return 1
	else:
		return 0

def f5(x,y):
	prepositions = ['about','above','across','after','against','along','among','around','at','before','behind','below','beneath','beside','between','by','down','during','except','for','from','in','inside','instead of','into','like','near','of','off','on','onto','out of','outside','over','past','since','through','to','toward','under','underneath','until','up','upon','with','within','without']
	if x[5][2].lower() in prepositions and y=='IN':
		return 1
	else:
		return 0

def f6(x,y):
	# wh questions
	if(x[5][2][:2].lower() == 'wh' and y in ['WDT','WP','WRB']):
		return 1
	else:
		return 0

def f7(x,y):
	conjunctions = ['and','or','but','nor','so','for', 'yet']
	if x[5][2].lower() in conjunctions and y=='CC':
		return 1
	else:
		return 0

def f8(x,y):
	# check if no
	s = x[5][2].lower()
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
	if x[5][2].lower() in determiners and y=='DT':
		return 1
	else:
		return 0

def f10(x,y):
	# check for existential 'there'
	# current word is 'there' and next word is the one in list
	if x[5][2].lower() =='there' and x[5][3].lower() in ['is','was','were','has'] and y=='EX':
		return 1
	else:
		return 0

def f11(x,y):
	# personal pronoun
	pp = ['i', 'you', 'he', 'she', 'it', 'they', 'we']
	if x[5][2].lower() in pp and y=='PRP':
		return 1
	else:
		return 0

def f12(x,y):
	if x[5][2].lower() == 'to' and y=='TO':
		return 1
	else:
		return 0

def f13(x,y):
	# check if ascii, FW means foreign word
	s = x[5][2]
	if(not all(ord(c) < 128 for c in s) and y=='FW'):
		return 1
	else:
		return 0

def f14(x,y):
	# modal verb
	modal = ['can', 'could', 'may', 'might', 'must', 'ought', 'shall', 'should', 'will', 'would']
	if x[5][2].lower() in modal and y=='MD':
		return 1
	else:
		return 0

def get_features(x,y):
	# print [f(x,y) for f in features]
	return [f(x,y) for f in features]

def gradient_preprocess1(input_data,tags):
	print 'Preprocessing for gradient'
	global all_input_features, all_input_tag_combination_features
	all_input_tag_combination_features = {}

	for i,x in enumerate(input_data):
		for y in tags:
			t = (x['i'],x['index'],x['t_minus_one'],x['t_minus_two'],x['tag'],x['wn'])
			features = all_input_tag_combination_features.get(y, [])
			current_input_tag_features = get_features(t, y)
			features.append(current_input_tag_features)
			all_input_tag_combination_features[y] = features
			if (x['tag'] == y):
				all_input_features.append(current_input_tag_features)

	# import pickle
	# pickle.dump(all_input_tag_combination_features,open('file_py','w'))
	# sys.exit()

	for k, v in all_input_tag_combination_features.items():
		all_input_tag_combination_features[k] = numpy.array(v)

	all_input_features_broadcast = sc.broadcast(numpy.array(all_input_features))
	all_input_tag_combination_features = sc.broadcast(all_input_tag_combination_features)
	print 'Done'
	return

def cost(params):
	num_calls_cost+=1
	t1 = time()
	def helper(datapoint):
		# sum_sqr_params = sum([p * p for p in params]) # for regularization
		# reg_term = 0.5 * self.reg * sum_sqr_params

		emperical = 0
		expected = 0

		dot_product = numpy.dot(get_features(datapoint, datapoint.tag), params)
		emperical += dot_product
		temp = 0
		for y in all_tags.value:
			dot_product = numpy.dot(get_features(datapoint, datapoint.tag), params)
			temp += math.exp(dot_product)
		expected += math.log(temp)
		return (expected, emperical)
		
	expected,emperical = distributed_input_data.map(helper).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]))
	cost = (expected - emperical)
	tot_time_cost+=time()-t1
	print '--------------------------------------------------'
	print cost, expected, emperical
	print '--------------------------------------------------'
	return cost

def cost1(params):
	# print '--------------------------------------------------'
	print params,'cost params'
	print '--------------------------------------------------'

	# param_b = sc.broadcast(params)
	global num_calls_cost,tot_time_cost
	num_calls_cost+=1
	t1 = time()
	def helper(datapoint):
		# sum_sqr_params = sum([p * p for p in params]) # for regularization
		# reg_term = 0.5 * self.reg * sum_sqr_params
		emperical_part = numpy.dot(get_features(datapoint, datapoint.tag), params)
		temp = 0
		for y in all_tags.value:
			dot_product = numpy.dot(get_features(datapoint, y), params)
			temp += math.exp(dot_product)
		expected_part = math.log(temp)
		return (expected_part, emperical_part)
		
	expected,emperical = distributed_input_data.map(helper).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]))
	cost = (expected - emperical)
	tot_time_cost+=time()-t1
	# print '--------------------------------------------------'
	print cost, expected, emperical
	print '--------------------------------------------------'
	return cost


def gradient1(params):
	print '--------------------------------------------------'
	print params,'params'
	print '--------------------------------------------------'
	num_calls_gradient+=1
	t1 = time()
	
	def helper(k):
		# reg_term = self.reg * params[k]
		reg_term = 0.0
		empirical = 0.0
		expected = 0.0
		for dx in all_input_features:
			empirical += dx[k]

		for i in range(size.value):
			denominator = 0.0
			for y in all_tags.value:
				feat = all_input_tag_combination_features.value[y][i]
				denominator += math.exp(numpy.dot(feat, params))
			mysum = 0.0 # exp value per example
			
			for y in all_tags.value: # for each tag compute the exp value
				fx_yprime = all_input_tag_combination_features.value[y][i] #self.get_feats(self.h_tuples[i][0], t)
				dot_vector = numpy.dot(numpy.array(fx_yprime), params)
				numerator = math.exp(dot_vector) 
				prob = numerator / denominator
				mysum += prob * float(fx_yprime[k])                    

			expected += mysum
		return (expected - empirical + reg_term)
	
	gradient = numpy.array(sc.parallelize(list(range(no_of_features.value))).map(helper).collect())
	tot_time_gradient += time()-t1
	print '--------------------------------------------------'
	print gradient,'gradient'
	print '--------------------------------------------------'
	return gradient
	
def gradient1_new(params):
	# print '--------------------------------------------------'
	# print params,'grad params'
	# print '--------------------------------------------------'
	global num_calls_gradient, tot_time_gradient
	num_calls_gradient += 1
	t1 = time()
	gradient = []
	

	for k in range(no_of_features.value): # vk is a m dimensional vector
		# reg_term = self.reg * params[k]
		reg_term = 0.0
		empirical = 0.0
		expected = 0.0
		for dx in all_input_features:
			empirical += dx[k]
		# print "emerpical",empirical
		def helper(datapoint):
			denominator = 0.0
			for y in all_tags.value:
				feat = all_input_tag_combination_features.value[y][datapoint.i]
				denominator += math.exp(numpy.dot(feat, params))
			mysum = 0.0 # exp value per example
			
			for y in all_tags.value: # for each tag compute the exp value
				fx_yprime = all_input_tag_combination_features.value[y][datapoint.i] #self.get_feats(self.h_tuples[i][0], t)

				dot_vector = numpy.dot(numpy.array(fx_yprime), params)
				numerator = math.exp(dot_vector) 
				prob = numerator / denominator
				mysum += prob * float(fx_yprime[k])                    
			return mysum
			print denominator, numerator,mysum,"dnm"
		expected = distributed_input_data.map(helper).reduce(lambda x,y:x+y)
		print empirical,expected
		gradient.append(expected - empirical + reg_term)
	tot_time_gradient += time()-t1
	print '--------------------------------------------------'
	# print gradient,'gradient'
	# print '--------------------------------------------------'
	return numpy.array(gradient)
# def gradient(params):
# 	for i in range(self.num_examples):
# 	def helper(datapoint):	
# 		denominator = 0.0
# 		for y in self.all_y:
# 			feat = self.all_data[y][i]
# 			denominator += math.exp(numpy.dot(feat, self.param))
		
# 		for y in self.all_y: # for each tag compute the exp value
# 			fx_yprime = self.all_data[y][i] #self.get_feats(self.h_tuples[i][0], t)

# 			dot_vector = numpy.dot(numpy.array(fx_yprime), self.param)
# 			numerator = math.exp(dot_vector) 
# 			prob = numerator / denominator
			
# 			for j,val in enumerate(all_sum):
# 				all_sum[j] += prob * fx_yprime[j]


if __name__ == '__main__':
	features = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10, f11, f12, f13,f14]
	# features = [f1,f2,f3]

	input_data,tags = create_input_dataset()
	distributed_input_data = sqlContext.jsonFile('data.json')
	gradient_preprocess1(input_data,list(set(tags)))
	# print distributed_input_data.show()
	
	all_tags = sc.broadcast(list(set(tags)))
	no_of_features = sc.broadcast(len(features))
	size = sc.broadcast(len(input_data))
	param = [0 for i in range(len(features))]
	# gradient1_new(param)
	# param = [1 for i in range(len(features))]
	# gradient1_new(param)
	dt1 = datetime.datetime.now()
	print 'before training: ', dt1
	params = mymin(cost1, param, method = 'L-BFGS-B', jac = gradient1_new, options = {'maxiter':100}) #, jac = self.gradient) # , options = {'maxiter':100}
	print params.x
	print params

	dt2 = datetime.datetime.now()
	print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()

	print 'For',num_calls_cost,'calls to cost total time taken is',tot_time_cost
	print 'Per call avg time taken is',tot_time_cost/num_calls_cost
	print 'For',num_calls_gradient,'calls to gradient total time taken is',tot_time_gradient
	print 'Per call avg time taken is',tot_time_gradient/num_calls_gradient

