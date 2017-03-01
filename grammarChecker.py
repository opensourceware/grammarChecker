import numpy as np
import math
import cPickle
import nltk
import optparse
from nltk.corpus import brown
import matplotlib.pyplot as plt

def count(word, train_sentences):
	count=0
	for sent in train_sentences:
		for w in sent:
			if w==word:
				count+=1
	return count

def bigram(pair):
	count = 0
	for sent in train_sentences:
		for i in range(len(sent)-1):
			bigram = sent[i]+' '+sent[i+1]
			if bigram==pair:
				count+=1
	return count


class BigramEM:
	"""EM algorithm for aggregate bigram model as described by Saul and Pereira, 1997.
	"""
	def __init__(self, V, C, B):
		self.class_size = C
		self.transitionMatrix = np.random.rand(C, V)
		self.emissionMatrix = np.random.rand(V, C)
		self.latent_posprob = np.random.rand(C, B)

	def EStep(self, word_to_id, bigram_to_id):
		bigrams = bigram_to_id.items()
		for bigram in bigrams:
			words = bigram[0].split(' ')
			w1_index = word_to_id[words[0]]
			w2_index = word_to_id[words[1]]
			bigram_index = bigram[1][0]
			#print bigram_index
			#print self.latent_posprob[:, bigram_index]
			self.latent_posprob[:, bigram_index] = self.findposprob(self.class_size, w1_index, w2_index)

	def MStep(self, word_to_id, bigram_to_id, train_sentences=[], expected_count=False):
		bigrams = bigram_to_id.items()
		bigrams = sorted(bigrams, key=self.getkey_wrt_w1)
		prev_word = bigrams[0][0].split(' ')[0]
		vec = [0.0]*self.class_size
		for bigram in bigrams:
			words = bigram[0].split(' ')
			w1_index = word_to_id[words[0]]
			w2_index = word_to_id[words[1]]
			bigram_index = bigram[1][0]
			bigram_freq = bigram[1][1]
			if words[0] == prev_word:
				for c_index in range(self.class_size):
					prob = bigram_freq*self.latent_posprob[c_index, bigram_index]
					vec[c_index] += prob
			else:
				#print prev_word
				normalizing_factor = sum(vec)
				self.transitionMatrix[:, word_to_id[prev_word]] = [prob/normalizing_factor for prob in vec]
				vec = [0.0]*self.class_size
				prev_word = words[0]
				for c_index in range(self.class_size):
					prob = bigram_freq*self.latent_posprob[c_index, bigram_index]
					vec[c_index] += prob

		bigrams = sorted(bigrams, key=self.getkey_wrt_w2)
		prev_word = bigrams[0][0].split(' ')[1]
		vec = [0.0]*self.class_size
		normalizing_factor = 0.0
		for bigram in bigrams:
			words = bigram[0].split(' ')
			w1_index = word_to_id[words[0]]
			w2_index = word_to_id[words[1]]
			bigram_index = bigram[1][0]
			bigram_freq = bigram[1][1]
			if words[1] == prev_word:
				for c_index in range(self.class_size):
					prob = bigram_freq*self.latent_posprob[c_index, bigram_index]
					vec[c_index] += prob
			else:
				#print prev_word
				self.emissionMatrix[word_to_id[prev_word], :] = vec
				vec = [0.0]*self.class_size
				prev_word = words[1]
				for c_index in range(self.class_size):
					prob = bigram_freq*self.latent_posprob[c_index, bigram_index]
					vec[c_index] += prob
		if expected_count:
			counter = 0
			for (word, index) in word_to_id.items():
				print 'Expected count of word '+word +' is: '+str(sum(self.emissionMatrix[index, :]))
				print 'Actual counts of word '+word +' is: '+str(count(word, train_sentences))
				counter+=1
				if counter == 5:
					break
		for c_index in range(self.class_size):
			normalizing_factor = sum(self.emissionMatrix[:, c_index])
			self.emissionMatrix[:, c_index] /= normalizing_factor

	def getkey_wrt_w2(self, item):
		return item[0].split(' ')[1]

	def getkey_wrt_w1(self, item):
		return item[0].split(' ')[0]

	def findposprob(self, C, w1_index, w2_index):
		posvec = []
		for c_index in range(C):
			prob = 0
			prob = self.emissionMatrix[w2_index, c_index] * self.transitionMatrix[c_index, w1_index]
			posvec.append(prob)
		normalizing_factor = sum(posvec)
		posvec = [prob/normalizing_factor for prob in posvec]
		return posvec

	def corpusLL(self, sents, word_to_id):
		TokLL = 0.0
		for sent in sents:
			for i in range(len(sent)-1):
				prob = np.dot(self.emissionMatrix[word_to_id[sent[i+1]], :], self.transitionMatrix[:, word_to_id[sent[i]]])
				TokLL+=math.log(prob)
		return TokLL/len(brown.words())

	def LL(self, sent, word_to_id):
		loglikelihood = 0.0
		for i in range(len(sent)-1):
			prob = np.dot(self.emissionMatrix[word_to_id[sent[i+1]], :], self.transitionMatrix[:, word_to_id[sent[i]]])
			loglikelihood += math.log(prob)
		return loglikelihood


def create_mapping(sents):
	unigramDict = {}
	bigramDict = {}
	#bigramhash_w1 = {}
	unigram_index = 0
	bigram_index = 0
	for sent in sents:
		i = -1
		sent.append('END')
		for i in range(len(sent)-1):
			#word1 = sent[i]
			#word2 = sent[i+1]
			pair = sent[i]+' '+sent[i+1]
			if pair not in bigramDict:
				bigramDict[pair] = (bigram_index, 1)
				bigram_index+=1
			else:
				bigramDict[pair] = (bigramDict[pair][0], bigramDict[pair][1]+1)
			if sent[i] not in unigramDict:
				unigramDict[sent[i]] = unigram_index
				unigram_index+=1
		#print sent
		if sent[i+1] not in unigramDict:
			unigramDict[sent[i+1]] = unigram_index
			unigram_index+=1
	return (unigramDict, bigramDict)

def unigramMLE(sent):
	sent_prob = 0.0
	for word in sent:
		prob = count(word)
		sent_prob += math.log(prob)
	sent_prob -= len(sent)*math.log(len(brown.words())+len(brown.sents()))
	return sent_prob

def bigramMLE(sent):
	sent_prob = 0.0
	for i in range(len(sent)-1):
		pair = sent[i]+' '+sent[i+1]
		count = bigram(pair)
		if count==0.0:
			prob = 0.0
			sent_prob += prob
		else:
			sent_prob += (prob/normalizing_factor)
	return sent_prob

if __name__ == "__main__":

	sents = brown.sents()
	train_sentences = []
	for sent in sents:
		for ind, item in enumerate(sent):
			sent[ind] = item.lower()
		sent.append('END')
		train_sentences.append(sent)
	(word_to_id, bigram_to_id) = create_mapping(train_sentences)
	vocab_size = len(word_to_id)
	bigram_size = len(bigram_to_id)

	optparser = optparse.OptionParser()
	optparser.add_option(
	    "-c", "--latentclass_size", default=5,
	    help="Number of latent classes"
	)
	optparser.add_option(
	    "-l", "--iterations", default=20,
	    help="Number of training iterations"
	)
	opts = optparser.parse_args()[0]

	latentclass_size = opts.latentclass_size
	iterations = opts.iterations

	EMPredictor = BigramEM(vocab_size, latentclass_size, bigram_size)
	for i in range(iterations):
		EMPredictor.EStep(word_to_id, bigram_to_id)
		EMPredictor.MStep(word_to_id, bigram_to_id)
		#TokLL = EMPredictor.corpusLL(train_sentences, word_to_id)

	while True:
		sent1 = raw_input('Enter the first sentence: ')
		sent1_tokens = sent1.split()
		sent2 = raw_input('Enter the second sentence: ')
		sent2_tokens = sent2.split()
		prob1 = EMPredictor.LL(sent1_tokens, word_to_id)
		prob2 = EMPredictor.LL(sent2_tokens, word_to_id)
		if prob1<prob2:
			print sent1+'\nis grammatically more correct than \n'+ sent2
		else:
			print sent2+'\nis grammatically more correct than \n'+ sent1