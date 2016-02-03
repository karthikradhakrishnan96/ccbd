import numpy
import math
import itertools
import datetime
from scipy.optimize import minimize as mymin 
from nltk.tokenize import word_tokenize as wt
from time import time
from  pdb import set_trace
###########################
# leave out gradient for now.
##########################

class MEMM(object):
	"""docstring for MEMM"""
	def __init__(self, X, Y, all_y,param, text,feature_functions, reg):
		self.X = X 	# input set
		self.Y = Y 	# label for input set
		self.func = feature_functions 	# array of feature functions
		self.all_y = all_y # all possible outputs. basically set(Y)
		self.param = param # parameter vector. this is what we get after training
		self.reg = reg # regularization term. dont bother about it for now. i'll explain in detail later.
		self.dim = 0
		self.text = text
		self.num_calls_cost = 0
		self.num_calls_gradient = 0
		self.tot_time_cost = 0
		self.tot_time_gradient = 0
		self.preprocess()
		#print 'Preprocessing for gradient'
		self.dataset = []
		self.all_data = {}
		#set_trace()
		for i,x in enumerate(self.X):
			for y in self.all_y:
				feats = self.all_data.get(y, [])
				val = self.get_features(x, y)
				feats.append(val)
				self.all_data[y] = feats
				if (self.Y[i] == y):
					self.dataset.append(val)
					#print 'X:',x['wn'][2],'Y:',y
					#print val
		#print '######DATASET:',len(self.dataset),'-',len(self.dataset[0]),'#########ALL Data',len(self.all_data),'-',len(self.all_data[all_y[0]]),'-',self.all_data[all_y[0]]
		for k, v in self.all_data.items():
			self.all_data[k] = numpy.array(v)
		self.dataset = numpy.array(self.dataset)

		self.num_examples = len(self.X)
		#print self.num_examples,len(self.all_y)
		#print 'Done'
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
		self.dim = 3#len(list(self.word_index))*len(self.all_y)
		self.param = [0.0 for i in range(self.dim)]
		#print len(list(self.all_words))
		#print len(self.all_y)
		#print self.all_y

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
		#print 'X',x['wn'][2],'Y',y
		feat_vec = [0.0 for _ in range(self.dim)]
		present_word = x['wn'][2].strip()
		first_char = ord(present_word[0])
		if first_char >= 65 and first_char <=90:
			feat_vec[0] = 1
		if len(present_word) >=3:
			feat_vec[1] = 1
		if len(y) <3 :
			feat_vec[2] = 1
		#print feat_vec
		return feat_vec
	
	def cost(self, params):
		'''
		the cost function is the derivative of the p_y_given_x probability distribution summed over all inputs.
		i'll send the formula in hangout. 

		'''
		
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
		#print 'took',time()-t1,'s'
		self.tot_time_cost+=time()-t1
		return cost

	def train(self):
		dt1 = time()
		#print 'before training: ', dt1
		# this is the optimization function. spark already has this one. we'll use that.
		# it takes cost, parameter vector and modifies the parameter vector. the process continues
		# untill training is complete
		self.preprocess()
		params = mymin(self.cost, self.param, method = 'L-BFGS-B',jac = self.gradient, options = {'maxiter':1}) #, jac = self.gradient) # , options = {'maxiter':100}
		# self.gradient([0,0,0])
		#self.param = params.x
		# self.gradient(self.param)
		print params
		dt2 = time()

	def gradient(self, params):
		#set_trace()
		self.num_calls_gradient+=1
		t1 = time()
		gradient= [x-0.05 for x in self.param]
		print params
		gradient = []

		#print "dim",self.dim,"num eg",self.num_examples,"all_y",len(self.all_y)

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
			print "em",empirical,"r",reg_term,"ex",expected
			#print "1:",expected +reg_term , ":2:" , expected - empirical
			#print "gradient",(expected - empirical + reg_term)
		print gradient
		#print numpy.array(gradient)
		#print 'took',time()-t1,'s'
		self.tot_time_gradient+=time()-t1
		return numpy.array(gradient)
