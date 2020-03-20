import nltk
import random
import time
import numpy as np

training_percent = 0.6

def get_features(transcript, word_features):
    words = set(transcript)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

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
    divider = int(len(final_transcripts) * training_percent)
    training_set = final_transcripts[:divider]
    testing_set = final_transcripts[divider:]

    # train and run classifier
    clf = nltk.NaiveBayesClassifier.train(training_set)
    print('\nNaive Bayes Classifier accuracy: ', nltk.classify.accuracy(clf, testing_set), '\n')

    # display other relevent data
    clf.show_most_informative_features(20)



def main():
    print('Loading transcripts...')
    transcripts = np.load('transcripts.npy', allow_pickle=True)
    random.shuffle(transcripts)

    start = time.time()
    print('Running Naive Bayes Classifier...')
    naive_bayes(transcripts)
    end = time.time()

    print('Completed classification in: ', end - start)


if __name__ == '__main__':
    main()