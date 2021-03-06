import numpy
import math
import itertools
import datetime
from time import time
from scipy.optimize import minimize as mymin 
from nltk.tokenize import word_tokenize as wt

import random

cost_no_of_calls = 0
grad_no_of_calls = 0

class MEMM(object):
	"""docstring for MEMM"""
	def __init__(self, X, Y, all_y,param, text,feature_functions, reg):
		self.X = X 	# input set
		self.Y = Y 	# label for input set
		self.func = feature_functions 	# array of feature functions
		self.all_y = all_y # all possible outputs. basically set(Y)
		self.reg = 0.0 # regularization term. dont bother about it for now. i'll explain in detail later.
		self.dim = len(feature_functions)
		self.text = text
		self.param = [0.0 for i in range(self.dim)]

		import random
		self.temp = [random.random() for i in range(13)]
		self.num_calls_cost = 0
		self.num_calls_gradient = 0
		self.tot_time_cost = 0
		self.tot_time_gradient = 0

		# self.preprocess()
		print 'Preprocessing for gradient'
		self.dataset = []
		self.all_data = {}
		for i,x in enumerate(self.X):
			for y in self.all_y:
				feats = self.all_data.get(y, [])
				val = self.get_features(x, y)
				feats.append(val)
				self.all_data[y] = feats
				if (self.Y[i] == y):
					self.dataset.append(val)
		for k, v in self.all_data.items():
			self.all_data[k] = numpy.array(v)
		self.dataset = numpy.array(self.dataset)
		# print self.dataset
		self.num_examples = len(self.X)
		# print self.num_examples,len(self.all_y)
		# print self.dataset[:10]
		print 'Done'
		return
	
	def preprocess(self):
		word_count = {}
		self.all_words = set()
		self.word_index = {}
		self.tag_index = {}
		index = 0
		for tag in self.all_y:
			self.tag_index[tag] = index
			index += 1

		for sent in self.text:
			for word,tag in sent:
				word_count[word] = word_count.get(word, 0) + 1
		index = 0
		for k,v in word_count.iteritems():
			if(v>0):
				self.all_words.add(k)
				self.word_index[k] = index
				index += 1
		self.dim = len(list(self.word_index))*len(self.all_y)
		self.param = [0.0 for i in range(self.dim)]
		# print len(list(self.all_words))
		# print len(self.all_y)
		# print self.all_y

	def p_y_given_x(self,x,y):
		''' This is the probability distribution.
		x,y are inputs given by the user.
		y belongs to a set of possible outputs self.all_y
		dot product of feature vector(refer get_features) and the parameter vector is computed for the numerator
		same is repeated for self.all_y
		basically its (x,y)/(x,all_y)

	'''
		features = self.get_features(x, y)
		numerator = math.exp(numpy.dot(features, self.param))

		denominator = 0
		for y in self.all_y:
			features_temp = self.get_features(x, y)
			temp = math.exp(numpy.dot(features_temp, self.param))
			denominator += temp

		return numerator/denominator
		
	def get_features(self, x, y):
		# return self.temp
		return [f(x,y) for f in self.func]

		length = self.dim
		feat_vec = [0.0 for i in range(length)]
		punctuations = ['.', ',', ':',';','\"','\'','``','\'\'']
		word = x['wn'][2]
		i = self.word_index[word]
		j = self.tag_index[y]
		feat_vec[len(self.all_words)*j + i] = 1.0
		return feat_vec


	
	def cost(self, params):
		'''
		the cost function is the derivative of the p_y_given_x probability distribution summed over all inputs.
		i'll send the formula in hangout. 

		'''
		# print params,'cost'
		self.num_calls_cost+=1
		t1 = time()
		self.param = params
		sum_sqr_params = sum([p * p for p in params]) # for regularization
		reg_term = 0.5 * self.reg * sum_sqr_params

		emperical = 0
		expected = 0
		for x,y in itertools.izip(self.X, self.Y):
			dp = numpy.dot(self.get_features(x, y), self.param)
			emperical += dp
			temp = 0
			for y in self.all_y:
				dp = numpy.dot(self.param,self.get_features(x, y))
				temp += math.exp(dp)
			expected += math.log(temp)
		cost = (expected - emperical) + reg_term
		self.tot_time_cost+=time()-t1
		# print self.param
		print cost,emperical,expected,reg_term
		return cost

	def train(self):
		dt1 = datetime.datetime.now()
		print 'before training: ', dt1
		# self.preprocess()
		# self.gradient2(self.param)
		params = mymin(self.cost, self.param, method = 'L-BFGS-B', jac = self.gradient1, options = {'maxiter':110}) #, jac = self.gradient) # , options = {'maxiter':100}
		self.param = params.x
		print self.param
		# import random
		# param = [random.random() for i in range(self.dim)]
		# print self.gradient1(param) == self.gradient2(param)
		# dt2 = datetime.datetime.now()
		# print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()
		# dt1 = datetime.datetime.now()
		# self.gradient2(self.param)
		dt2 = datetime.datetime.now()
		print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()
		print cost_no_of_calls, grad_no_of_calls

	def gradient(self, params):
		self.param = params
		gradient = []
		for k in range(self.dim): # vk is a m dimensional vector
			reg_term = self.reg * params[k]
			empirical = 0.0
			expected = 0.0
			for dx in self.dataset:
				empirical += dx[k]
			for i in range(self.num_examples):
				mysum = 0.0 # exp value per example
				for y in self.all_y: # for each tag compute the exp value
					fx_yprime = self.all_data[y][i] #self.get_feats(self.h_tuples[i][0], t)

					# --------------------------------------------------------
					# computation of p_y_given_x
					normalizer = 0.0
					dot_vector = numpy.dot(numpy.array(fx_yprime), self.param)
					for y1 in self.all_y:
						feat = self.all_data[y1][i]
						dp = numpy.dot(feat, self.param)
						if dp == 0:
							normalizer += 1.0
						else:
							normalizer += math.exp(dp)
					if dot_vector == 0:
						val = 1.0
					else:
						val = math.exp(dot_vector) # 
					prob = float(val) / normalizer
					# --------------------------------------------------------
					
					mysum += prob * float(fx_yprime[k])                    
				expected += mysum
			gradient.append(expected - empirical + reg_term)
		print numpy.array(gradient)
		return numpy.array(gradient)

	def gradient1(self, params):
		self.num_calls_gradient+=1
		t1 = time()
		self.param = params
		gradient = []
		for k in range(self.dim): # vk is a m dimensional vector
			reg_term = self.reg * params[k]
			empirical = 0.0
			expected = 0.0
			for dx in self.dataset:
				empirical += dx[k]
			print "emerpical",empirical
			for i in range(self.num_examples):
				denominator = 0.0
				for y in self.all_y:
					feat = self.all_data[y][i]
					denominator += math.exp(numpy.dot(feat, self.param))
				mysum = 0.0 # exp value per example
				
				for y in self.all_y: # for each tag compute the exp value
					fx_yprime = self.all_data[y][i] #self.get_feats(self.h_tuples[i][0], t)

					dot_vector = numpy.dot(numpy.array(fx_yprime), self.param)
					numerator = math.exp(dot_vector) 
					prob = numerator / denominator
					mysum += prob * float(fx_yprime[k])                    

				expected += mysum
			gradient.append(expected - empirical + reg_term)
		print expected
		print numpy.array(gradient)
		return numpy.array(gradient)

	def gradient2(self, params):
		# print params,'gradient'
		self.num_calls_gradient+=1
		t1 = time()
		self.param = params
		gradient = [0.0 for i in range(self.dim)]
		all_sum = [0.0 for i in range(self.dim)]

		for i in range(self.num_examples):
			denominator = 0.0
			for y in self.all_y:
				feat = self.all_data[y][i]
				denominator += math.exp(numpy.dot(feat, self.param))
			
			for y in self.all_y: # for each tag compute the exp value
				fx_yprime = self.all_data[y][i] #self.get_feats(self.h_tuples[i][0], t)

				dot_vector = numpy.dot(numpy.array(fx_yprime), self.param)
				numerator = math.exp(dot_vector) 
				prob = numerator / denominator
				
				for j,val in enumerate(all_sum):
					all_sum[j] += prob * fx_yprime[j]

		# print all_sum,'emperical'
		for k in range(self.dim): # vk is a m dimensional vector
			# reg_term = self.reg * params[k]
			emperical = 0.0
			for dx in self.dataset:
					# print len(dx)
					emperical += dx[k]
			gradient[k] = all_sum[k] - emperical
		print gradient
		self.tot_time_gradient+=time()-t1
		return numpy.array(gradient)


