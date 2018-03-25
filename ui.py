import spam_detector as sd
import text_preprocessor as tp
import glob
import nltk

NBclassifier = ''
MNBclassifier = ''
frequent_words = []
all_words = []

spam_dir = input('Enter spam dataset directory: ')
ham_dir = input('Enter ham dataset directory: ')

NBclassifier, NBaccuracy, MNBclassifier, MNBaccuracy, LRclassifier, LRaccuracy, SVMclassifier, SVMaccuracy, all_words, frequent_words = sd.train_classifier(spam_dir, ham_dir)
#NBclassifier, NBaccuracy, MNBclassifier, MNBaccuracy, all_words = sd.train_classifier(spam_dir, ham_dir)

all_words = list(all_words)

print('Multinomial Naive Bayes accuracy: ', MNBaccuracy)
print('Top 20 most informative features')
NBclassifier.show_most_informative_features(20)

while(True):
	
	email_folder = input('Enter evaluation foldername: ')
	filenames = glob.glob(email_folder + '\*.txt')

	print('\n\n')
	for email_filename in filenames:
		email = tp.load_email(email_filename)

		norm_features = []
		for word in email:
			norm_features.append(word.lower())
			all_words.append(word.lower())

		features = sd.find_features(norm_features, all_words, frequent_words)
		
		MNBresult = MNBclassifier.classify(features)
		LRresult = LRclassifier.classify(features)
		SVMresult = SVMclassifier.classify(features)

		print(email_filename)
		print('Multinomial Naive Bayes: ' + MNBresult)
		print('Logistic Regression: ' + LRresult)
		print('Support Vector Machine: ' + SVMresult)
		print('\n')

