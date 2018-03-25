import text_preprocessor as tp
import glob
import nltk
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
import random

#global variables
#emails = []	#list of tuples containing unprocessed data sets
#all_words = {}	#list containing all global words
#frequent_words = [] #list containing all frequent words
#classifier = object()

#collect all words from all emails
def collect_words(emails):
	all_words = []
	for email in emails:
		for word in email[0]:
			all_words.append(word.lower())
	
	all_words = nltk.FreqDist(all_words)

	frequent_words = list(all_words.keys())[:3000]

	print('total collected words: ', len(all_words))

	return all_words, frequent_words

def find_features(email, all_words, frequent_words):
	words = set(email)
	features = {}
	
	for word in frequent_words:
		features[word] = (word in words)
	
	return features

def train_classifier(spam_filefolder, ham_filefolder):
	
	spam_filenames = glob.glob(spam_filefolder + '\*.txt')
	ham_filenames = glob.glob(ham_filefolder + '\*.txt')
	
	emails = tp.load_emails(spam_filenames, ham_filenames)
	all_words, frequent_words = collect_words(emails)
	
	print('Extracting Feature Sets...')
	feature_sets = [(find_features(email, all_words, frequent_words), category) for (email, category) in emails]
	training_set = feature_sets[:int(len(feature_sets)*.60)]
	testing_set = feature_sets[int(len(feature_sets)*.60):]
	

	NBclassifier = nltk.NaiveBayesClassifier.train(training_set)
	NBaccuracy = nltk.classify.accuracy(NBclassifier, testing_set)

	print('Training Multinomial Naive Bayes Model...')
	#Multinomial Naive Bayes Classifier
	MNBclassifier = SklearnClassifier(MultinomialNB())
	MNBclassifier.train(training_set)
	MNBaccuracy = nltk.classify.accuracy(MNBclassifier, testing_set)

	print('Training Logistic Regression Model...')
	#Logistic Regression Classifier
	LRclassifier = SklearnClassifier(LogisticRegression())
	LRclassifier.train(training_set)
	LRaccuracy = nltk.classify.accuracy(LRclassifier, testing_set)

	print('Training Support Vector Machine...')
	#Support Vector Machine Classifier
	SVMclassifier = SklearnClassifier(SVC())
	SVMclassifier.train(training_set)
	SVMaccuracy = nltk.classify.accuracy(SVMclassifier, testing_set)

	#return NBclassifier, NBaccuracy, MNBclassifier, MNBaccuracy, all_words
	return NBclassifier, NBaccuracy, MNBclassifier, MNBaccuracy, LRclassifier, LRaccuracy, SVMclassifier, SVMaccuracy, all_words, frequent_words

def save_classifier(classifier):
	clf_file = open('naive_bayes_clf.pickle', 'wb')
	pickle.dump(classifier, clf_file)
	clf_file.close()
	print('Classifier saved to file: naive_bayes_clf.pickle')

def load_classifier():
	clf_file = open('naive_bayes_clf.pickle', 'rb')
	classifier = pickle.load(clf_file)
	clf_file.close()
	print('naive_baye_clf.pickle loaded to classifier')
	return classifier



