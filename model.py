"""
This is my docstring
"""

# nltk.download()
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union  # these are state union addresses
from nltk.tokenize import PunktSentenceTokenizer  # unsupervised ML tokenizer
from nltk.stem import WordNetLemmatizer

# using tokenizers ---------------------------------------------------

EXAMPLE_TEXT = "hello mr.smith, how are you doing today? " \
               "the weather is great and python is awesome. " \
               "the sky is blue. you should not eat cardboard"
# print(sent_tokenize(EXAMPLE_TEXT))

# ['hello mr.smith, how are you doing today?',
# 'the weather is great and python is awesome.',
# 'the sky is blue.', 'you should not eat cardboard']

# print(word_tokenize(EXAMPLE_TEXT))
# for i in word_tokenize(EXAMPLE_TEXT):
#     print(i)

# using stopwords ---------------------------------------------------

stop_words = set(stopwords.words("english"))
print(stop_words)
words = word_tokenize(EXAMPLE_TEXT)
# filtered_sentence = []
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
# print(filtered_sentence)

filtered_sentence = [w for w in words if not w in stop_words]
# print(filtered_sentence)

# stemming ---------------------------------------------------

ps = PorterStemmer()
example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
for w in example_words:
    print(ps.stem(w))
new_text = "it is very important to be pythonly while you are pythoning with python. " \
           "All pythoners have pythoned poorly at least once"
words = word_tokenize(new_text)
# for w in words:
#     print(ps.stem(w))

# part of speech tagging ---------------------------------------------------


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)  # sentence tokenizer
tokenized = custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized:
            # each i is a sentence
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
# process_content()


# chunking (grouping of things) ---------------------------------------------------

def process_content_2():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))
# process_content_2()


# chinking (removing of things) ---------------------------------------------------

def process_content_3():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))
# process_content_3()

# named entity recognition ---------------------------------------------------

def process_content_4():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)  #,binary=True would just say of the chunk is a named entity
            namedEnt.draw()
    except Exception as e:
        print(str(e))

# process_content_4()

# lemmatizing ---------------------------------------------------
# similar to stemming but we end up with an actual word

lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("mice"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("better"))
# print(lemmatizer.lemmatize("better", pos="a"))  # a for adjective - actually gives a different answer
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("run"))  # run can actually be a noun. i.e. going on a run
# print(lemmatizer.lemmatize("run", pos="v")) # v for verb
# the default lemmatizing parameter is n which is noun. e.g. better is like a person who takes a bet

# corpora ---------------------------------------------------

print(nltk.__file__) # where the python stuff is located

from nltk.corpus import gutenberg

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])

# wordnet --
# can look up synonyms, antonyms, context, etc

from nltk.corpus import wordnet
syns = wordnet.synsets("program")
print(syns)
print(syns[0])
print(syns[0].lemmas())  # some lemmas for this are plan, pl
print(syns[0].name())  # synset
print(syns[0].lemmas()[0].name())  # just the word
print(syns[0].definition())  # definition
print(syns[0].examples())  # examples

synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for lemma in syn.lemmas():
        print("lemma:", lemma)
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))

# looking at similarity
w1 = wordnet.synset("ship.n.01") # n for noun, 1 for first noun
w2 = wordnet.synset("boat.n.01")
# compare teh semantic similarity o the two
print(w1.wup_similarity(w2))  # wu palmer? # gives a percentage of similarity

w1 = wordnet.synset("ship.n.01") # n for noun, 1 for first noun
w2 = wordnet.synset("car.n.01")
# compare teh semantic similarity o the two
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01") # n for noun, 1 for first noun
w2 = wordnet.synset("cat.n.01")
# compare teh semantic similarity o the two
print(w1.wup_similarity(w2))


w1 = wordnet.synset("ship.n.01") # n for noun, 1 for first noun
w2 = wordnet.synset("cactus.n.01")
# compare teh semantic similarity o the two
print(w1.wup_similarity(w2))

# text classification ---------------------------------------------------

# creating our own algorithm to classify things
# # eg. spam vs not span, positive vs negative

import nltk
import random
from nltk.corpus import movie_reviews

documents2 = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# or (they both give the same thing):

documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        print("category:", category)
        print("fileid:", fileid)
        documents.append([list(movie_reviews.words(fileid)), category])
        # words gives the words in that movie review

random.shuffle(documents)
# print(documents[1])
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower()) # add all words to the list, lowercase

all_words = nltk.FreqDist(all_words) # get it into a frequenyc distribution
print(all_words.most_common(n=15)) # note that punctuation is counted as a word as well
print(all_words["stupid"])

# words as features for learning ---------------------------------------------------
# continue using the movie reviews dataset to classify as positive or negative

# don't want to use all the words since there are a lot

word_features = list(all_words.keys())[:3000]  # just want to use the first 3000 words # top 3000 words

def find_features(document): # document will be in documents
    words = set(document) # document is just the list of word
    features = {}
    for w in word_features:
        features[w] = (w in words) # true or false # true if this wod is in this document
    return features

print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# naive bayes ---------------------------------------------------

training_set = featuresets[:1900]
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("naive bayes algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)))
classifier.show_most_informative_features(15) #15 most informative features

# save classifier in a pickle ---------------------------------------------------
import pickle

save_classifier = open("naivebayes.pickle","wb") # save the classifier
pickle.dump(classifier, save_classifier)
save_classifier.close()

# try using the pickle file instead.
classifier_f = open("naivebayes.pickle","rb") # use the classifier
classifier = pickle.load(classifier_f)
classifier_f.close()
print("naive bayes algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)))
classifier.show_most_informative_features(15) #15 most informative features


# include the scikit learn toolkit for ML ---------------------------------------------------

from nltk.classify .scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# multinomial

MNB_classifier = SklearnClassifier(MultinomialNB()) # can still use nltk, but use sklearn packages
MNB_classifier.train(training_set)
print("MNB_classifier accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set)))

# GaussianNB_classifier = SklearnClassifier(GaussianNB()) # can still use nltk, but use sklearn packages
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB()) # can still use nltk, but use sklearn packages
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)))

from sklearn.linear_model import LogisticRegression, SGDClassifier # more things from sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression()) # can still use nltk, but use sklearn packages
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier()) # can still use nltk, but use sklearn packages
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)))

SVC_classifier = SklearnClassifier(SVC()) # can still use nltk, but use sklearn packages
SVC_classifier.train(training_set)
print("SVC_classifier accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set)))

LinearSVC_classifier = SklearnClassifier(LinearSVC()) # can still use nltk, but use sklearn packages
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

NuSVC_classifier = SklearnClassifier(NuSVC()) # can still use nltk, but use sklearn packages
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)))
