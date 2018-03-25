from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import text_preprocessor
import numpy
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def preprocess_reviews(rev_folder, nrev_folder):
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer = word_tokenize,
                                 max_features = 5000)
    dataset = text_preprocessor.load_emails(rev_folder, nrev_folder)
    random.shuffle(dataset)

    return dataset

def train_classifiers(dataset):
    train_data = dataset[:int(len(dataset)*.7)]
    test_data = dataset[int(len(dataset)*.7):]

    train_texts = [text for text, label, filename in train_data]
    train_labels = [label for text, label, filename in train_data]

    test_texts = [text for text, label, filename in test_data]
    test_labels = [label for text, label, filename in test_data]

    print('Vectorizing Documents...')
    vectorizer = CountVectorizer(analyzer = 'word', tokenizer = word_tokenize, max_features = 5000)
    train_features = vectorizer.fit_transform(train_texts)
    train_features = train_features.toarray()
    test_features = vectorizer.transform(test_texts)
    test_features = test_features.toarray()

    print('Training Multinomial Naive Bayes Classifier...')
    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(train_features, train_labels)
    mnb_predictions = mnb_classifier.predict(test_features)
    mnb_accuracy = accuracy_score(test_labels, mnb_predictions)

    print('Training Logistic Regression Classifier...')
    lr_classifier = LogisticRegression()
    lr_classifier.fit(train_features, train_labels)
    lr_predictions = lr_classifier.predict(test_features)
    lr_accuracy = accuracy_score(test_labels, lr_predictions)

    print('Training Support Vector Machine Classifier...')
    svm_classifier = SVC()
    svm_classifier.fit(train_features, train_labels)
    svm_predictions = svm_classifier.predict(test_features)
    svm_accuracy = accuracy_score(test_labels, svm_predictions)

    return mnb_classifier, lr_classifier, svm_classifier, mnb_accuracy, lr_accuracy, svm_accuracy, vectorizer

def evaluate_set(dataset, mnb_classifier, lr_classifier, svm_classifier, vectorizer):
    texts = [text for text, label, filename in dataset]
    labels = [label for text, label, filename in dataset]
    filenames = [label for text, label, filename in dataset]

    text_features = vectorizer.transform(texts)
    text_features = text_features.toarray()

    mnb_predictions = mnb_classifier.predict(text_features)
    lr_predictions = lr_classifier.predict(text_features)
    svm_predictions = svm_classifier.predict(text_features)

    print('True Label\t\tMNB Prediction\t\tLR Prediction\t\tSVM Prediction')
    for i in range(len(texts)):
        print(labels[i] + '\t\t' + mnb_predictions[i] + '\t\t' + lr_predictions[i] + '\t\t' + svm_predictions[i])

    mnb_a = accuracy_score(labels, mnb_predictions)
    lr_a = accuracy_score(labels, lr_predictions)
    svm_a = accuracy_score(labels, svm_predictions)

    return mnb_a, lr_a, svm_a



