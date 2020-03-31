import nltk
import random
import time
import numpy as np
import os 

training_percent = 0.6

def get_features(transcript, word_features):
    words = set(transcript)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

def divide_data(transcripts):
    divider = int(len(transcripts) * training_percent)
    return [transcripts[:divider], transcripts[divider:]]

def naive_bayes(transcript_data):
    all_words = []
    word_features = []

    # get all words from transcripts
    for row in transcript_data:
        # cut off category
        transcript = row[:-1]
        for word in transcript:
            all_words.append(word)

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())
    final_transcripts = [(get_features(row[:-1], word_features), row[-1]) for row in transcript_data]

    # divide amongst data
    training_set, testing_set = divide_data(final_transcripts)

    # train and run classifier
    clf = nltk.NaiveBayesClassifier.train(training_set)
    print('\nNaive Bayes Classifier accuracy: ', nltk.classify.accuracy(clf, testing_set), '\n')

    # display other relevent data
    clf.show_most_informative_features(20)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
def nlp_svm(transcript_data):
    # format transcripts for count vectorizer
    formatted_transcripts = []
    labels = []
    for transcript in transcript_data:
        labels.append(transcript[-1])
        transcript = transcript[:-1]
        ft = ' '.join(str(word) for word in transcript)
        formatted_transcripts.append(ft)

    train_data, test_data = divide_data(formatted_transcripts)
    train_labels, test_labels = divide_data(labels)

    clf = Pipeline( [ ('vectorize', CountVectorizer()), 
                      ('tfidf', TfidfTransformer()), 
                      ('svm-clf', SGDClassifier(loss='hinge', alpha=1e-3)) ] )
    clf = clf.fit(train_data, train_labels)
    
    predicted_svm = clf.predict(test_data)
    accuracy = np.mean(predicted_svm == test_labels)
    print('SVM Classifier Accuracy:', accuracy)


def main():
    print('Loading transcripts...')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    transcripts = np.load(dir_path + '\\transcripts.npy', allow_pickle=True)
    random.shuffle(transcripts)

    start = time.time()
    print('Running Naive Bayes Classifier...')
    naive_bayes(transcripts)
    end = time.time()

    start = time.time()
    print('Running SVM Classifier...')
    nlp_svm(transcripts)
    end = time.time()

    print('Completed classification in: ', end - start)


if __name__ == '__main__':
    main()