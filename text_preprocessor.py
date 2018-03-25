from nltk.tokenize import word_tokenize
import random

def load_emails(arr_spam_filenames, arr_ham_filenames):
	
	emails = []	#array of tuples each tuple containing a list of tokens and a category
	all_words = []

	#load spam emails
	ctr = 0;
	dec_err = 0;
	for spam_filename in arr_spam_filenames:
		email = ''
		try:	
			for line in open(spam_filename, 'r'):
				email = email + ' ' + line
			tokens = word_tokenize(email) #tokenize each email
			emails.append((tokens, 'Spam')) #add to dataset
			print(ctr, ' emails encoded')
			ctr = ctr + 1
		except UnicodeDecodeError:
			dec_err = dec_err + 1
	#load ham emails
	for ham_filename in arr_ham_filenames:
		email = ''
		try:	
			for line in open(ham_filename, 'r'):
				email = email + ' ' + line
			tokens = word_tokenize(email) #tokenize each email
			emails.append((tokens, 'Ham')) #add to dataset\
			print(ctr, ' emails encoded')
			ctr = ctr + 1
		except UnicodeDecodeError:
			dec_err = dec_err + 1

	print('\n', ctr, 'emails encoded\n', dec_err, 'encoding error')

	#shuffle dataset
	random.shuffle(emails)
	return emails

def load_email(email_filename):
	
	email = ''
	
	for line in open(email_filename, 'r'):
		email = email + line
	
	tokens = word_tokenize(email)
	
	return email