import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
import os
import pandas as pd
from keras.models import model_from_json
import time #  for time stamp on file names
#
# DATA_PATH = os.getcwd() + "/data"
# TRAIN_FILE = os.path.join(DATA_PATH, "train.txt")
# DEV_FILE = os.path.join(DATA_PATH, "dev")
# TEST_FILE = os.path.join(DATA_PATH, "test")
# WORD_VOCAB_FILE = os.path.join(DATA_PATH, "vocabulary")

f = open(TRAIN_FILE, "r")
train_content = (f.read())
f.close()

from nltk import corpus
from nltk.corpus import brown
# nltk.corpus.brown.sents()

UNK = "UNK"
NUM = "NUM"
sentence_num = []
word_num = []

puncs = '[ ,.?!:;\-\'\"``]'

all_df = pd.DataFrame()
all_df["words"] = nltk.corpus.brown.words()  # checked that these two are the same length
all_df["words"] = all_df['words'].str.lower()
all_df["pos"] = [tag[1] for tag in nltk.corpus.brown.tagged_words()]  # checked that these two are the same length
all_df["puncs_1_next"] =  all_df["words"].shift(-1) # so that if a punctuation comes after, we know
all_df["puncs_2_next"] =  all_df["words"].shift(-2)  # (sometimes there are combos of punctuation)

# all_df[all_df.words.isin(list(puncs))]
list_puncs = list(puncs) + ["``", "\'\'"]  # ",``", "\'\'."] <- these wil just get combined
all_df = all_df[~all_df.words.isin(list_puncs)]  # remove them
all_df.loc[~all_df['puncs_1_next'].isin(list_puncs), 'puncs_1_next'] = "SPACE"  #
all_df.loc[~all_df['puncs_2_next'].isin(list_puncs), 'puncs_2_next'] = "SPACE"  #
all_df["punc_next"] = all_df.apply(lambda row: row.puncs_1_next +
                                               (row.puncs_2_next if (row.puncs_2_next in list_puncs and
                                                                     row.puncs_1_next in list_puncs)else ""), axis=1)
all_df = all_df.drop(['puncs_1_next', 'puncs_2_next'], axis=1)
all_df = all_df.reset_index(drop=True)

# get chunks in the data - can combine sentences - give random numbers to them
indices_newsent = [i+1 for i, x in enumerate(list(all_df["punc_next"])) if any(j in x for j in [".", "!","?"])]
n_endpts = len(indices_newsent)
# make chunks - i.e. put some sentences together
indices_newchunk = indices_newsent[::2] # some lines are too long

too_long = [i for i,x in enumerate(list(np.diff(indices_newchunk))) if x>50] # places where the index is too big
add_endpts =[]
for i in too_long:
    diff = indices_newchunk[i+1] - indices_newchunk[i]
    split_into_pieces = diff//50 + 1
    size = diff/split_into_pieces
    for j in range(1,split_into_pieces):
        add_endpts.append(int(indices_newchunk[i] + j*size))
indices_newchunk = indices_newchunk + add_endpts
# max(np.diff(np.sort(indices_newchunk))) # maximum length is 50 now
all_df["newid"] = (all_df.index.isin(indices_newchunk)).cumsum()  # some lines are too long



##########
# deal with numbers
all_df.loc[all_df['words'].str.isdigit(), 'words'] = "9999"  # all numbers will be read as a number
##########
# deal with rare words
all_words2= nltk.FreqDist(all_df["words"])
MAX_VOCAB_SIZE = 10000
increasing_count = sorted(all_words2, key=all_words2.get)
word_features = increasing_count[-MAX_VOCAB_SIZE:-1]
all_df.loc[~all_df['words'].isin(word_features), 'words'] = "<UNK>"  # rare words will be <UNK>

# words, vocab, etc
# prepare the words
all_words = list(set(all_df["words"].values))
all_words.append("ENDPAD")
n_all_words = len(all_words); print(n_all_words)
tags = list(set(all_df["punc_next"].values))
n_tags = len(tags); print(n_tags) # 41 tags - that's quite a bit

##################### COMEBINE ALL CORPORA DATA FRAMES BY HERE




class ChunkGetter(object): # technically it is chunk getter, not sentence getter

    def __init__(self, data):
        self.n_chunk = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["words"].values.tolist(),
                                                           s["pos"].values.tolist(),
                                                           s["punc_next"].values.tolist())]
        self.grouped = self.data.groupby("newid").apply(agg_func)
        self.chunks = [s for s in self.grouped]

    def get_next(self):
        try:
            self.n_chunk += 1
            s = self.grouped[self.n_chunk]
            return s
        except:
            return None

getter = ChunkGetter(all_df)
chunks = getter.chunks

# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# plt.hist([len(s) for s in chunks], bins=50)
# plt.show()

max_len = 50
word2idx = {w: i for i, w in enumerate(all_words)} # create dictionaries
tag2idx = {t: i for i, t in enumerate(tags)}
# word2idx["mentor"]

#  map the chunks to a sequence of numbers and then pad the sequence.
from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in c] for c in chunks]  # 0 element is the word
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_all_words - 1)
# X[1]

# map the tags to a sequence of numbers and then pad the sequence.
y = [[tag2idx[w[2]] for w in c] for c in chunks]  # 2nd element is the tag
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["SPACE"])
# y[1]

# change the y labels to categorical
from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]

# split into train and test set
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

# BUILD AND FIT THE LSTM MODEL ##################
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_all_words, output_dim=50, input_length=max_len)(input)  # 50-dim embedding
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=10, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=3, validation_split=0.1, verbose=1)
hist = pd.DataFrame(history.history)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.show()

i = 20
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
# print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
true_sentence =""
predict_sentence = ""
counter =0
for w, pred in zip(X_te[i], p[0]):
    true_punc = tags[list(y_te[i][counter]).index(1)] # true punctuation
    counter += 1
    true_sentence += all_words[w] + " " + true_punc + " "

    predict_sentence += all_words[w] + " " + tags[pred] + " "
    # clean up
    true_sentence = true_sentence.replace("SPACE", " ")
    " ".join(true_sentence.split())

    predict_sentence = predict_sentence.replace("SPACE", " ")
    " ".join(predict_sentence.split())
    # print("{:15}: {}".format(all_words[w], tags[pred]))
print(true_sentence)
print(predict_sentence)

results = model.evaluate(X_te, np.array(y_te), batch_size=128)
print("test loss, test acc:", results)

from sklearn.metrics import classification_report
import numpy as np

Y_test = np.argmax(y_te, axis=1) # Convert one-hot to index
y_pred = np.argmax(model.predict(np.array([X_te])),axis=-1)
print(classification_report(Y_test, p))

####### EVALUATIONS
test_eval_true = []
test_eval_pred = []
for row in range(X_te.shape[0]):
    p = model.predict(np.array([X_te[row]])) #### MODEL, NOT LOADED MODEL
    p = np.argmax(p, axis=-1)
    for j in range(50):
        if all_words[X_te[row][j]] != "ENDPAD":
            true = list(y_te[row][j]).index(1)
            pred = p[0][j]
            test_eval_true.append(true)
            test_eval_pred.append(pred)

print(classification_report(test_eval_true, test_eval_pred))




############### SAVE THE MODEL
# serialize model to JSON
model_json = model.to_json()
with open("model_" + time.strftime("%Y%m%d-%H%M%S") + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_" + time.strftime("%Y%m%d-%H%M%S") + ".h5")
print("Saved model to disk")

### LATER, LOAD THE MODEL
# load json and create model
json_file = open('model_' + "20200701-202753" + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_" + "20200701-202753" + ".h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_te, y_te, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


####### EVALUATIONS
test_eval_true = []
test_eval_pred = []
for row in range(X_te.shape[0]):
    p = model.predict(np.array([X_te[row]]))
    p = np.argmax(p, axis=-1)
    for j in range(50):
        if all_words[X_te[row][j]] != "ENDPAD":
            pred = p[0][j]
            true = list(y_te[row][j]).index(1)
            test_eval_true.append(true)
            test_eval_pred.append(pred)
print(classification_report(test_eval_true, test_eval_pred))





#####

i = 1
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
# print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
true_sentence =""
predict_sentence = ""
counter =0
for w, pred in zip(X_te[i], p[0]):
    true_punc = tags[list(y_te[i][counter]).index(1)] # true punctuation
    counter += 1
    true_sentence += all_words[w] + " " + true_punc + " "

    predict_sentence += all_words[w] + " " + tags[pred] + " "
    # clean up
    true_sentence = true_sentence.replace("SPACE", " ")
    " ".join(true_sentence.split())

    predict_sentence = predict_sentence.replace("SPACE", " ")
    " ".join(predict_sentence.split())
    # print("{:15}: {}".format(all_words[w], tags[pred]))
print(true_sentence)
print(predict_sentence)




######## try a new sentence

def predict_new(mod=model, sent_play="hello"):
    predict_sentence = ""
    words_play = sent_play.split()
    words_play_vocab = [w if w in all_words else "<UNK>" for w in words_play]
    X_play = np.array([[word2idx[w] for w in words_play_vocab]])
    # pos_tags = nltk.pos_tag(words)
    X_play = pad_sequences(maxlen=max_len, sequences=X_play, padding="post", value=n_all_words - 1)
    prediction = mod.predict(X_play)
    prediction = np.argmax(prediction, axis=-1)
    for w, pred in zip(words_play, prediction[0]):
        predict_sentence += w + " " + tags[pred] + " "
        # clean up
        predict_sentence = predict_sentence.replace("SPACE", " ")
    " ".join(predict_sentence.split())
        # print("{:15}: {}".format(all_words[w], tags[pred]))
    print(predict_sentence)

predict_new(model, "hello my name is bob my favorite stuffed animal is a cat and i love food")
predict_new(model, "hello my name is vivian i live in toronto where the sun gibbers")
predict_new(model, "hi my name is edward and i am an expert at machine learning i watch birds in my spare time")
