"""
Punctuation Transcription
Author: Vivian Ngo
Date: July 2020
"""

import nltk
from nltk import corpus
from nltk.corpus import brown
from nltk.corpus import gutenberg
import numpy as np
import pandas as pd
from keras.models import model_from_json
import time  # for time stamp on file names
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Punc_data(object):
    def __init__(self, corpora=[nltk.corpus.brown, nltk.corpus.gutenberg]):
        self.corpora = corpora
        self.puncs = '[ ,.?!:;\-\'\"``]'
        self.list_puncs = list(self.puncs) + ["``", "\'\'"]
        self.PAD = "ENDPAD"
        self.UNK = "UNK"
        self.spacekey = "SPACE"
        self.numkey = "9999"
        self.MAX_VOCAB_SIZE = 10000
        self.MAX_CHUNK_SIZE = 50  # max word length per chunk
        self.df = None
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.vocab = None
        self.n_vocab = None
        self.tags = None
        self.n_tags = None
        self.word2idx = None
        self.tag2idx = None

    def preprocess_data(self):
        """
        Preprocess the data so that we can use it for training and testing.
        Creates self.data_df, self.X_tr, self.y_tr, self.X_te, and self.y_te (training and testing datasets).
        :param corpora: a list of the nltk corpuses that we will use in training and testing
        :type corpora: list
        :return: None
        :rtype: None
        """
        # gather the data from specified corpora into the required format
        df = pd.DataFrame(columns = ['words', 'pos', 'punc_next'])
        df.columns = ['words', 'pos', 'punc_next']
        for corp in self.corpora:
            temp_df = pd.DataFrame()
            temp_df["words"] = corp.words()
            temp_df["words"].str.lower()  # remove capitalization
            # temp_df["pos"] = [tag[1] for tag in corpora.tagged_words()]
            temp_df["puncs_1_next"] = temp_df["words"].shift(-1)
            temp_df["puncs_2_next"] = temp_df["words"].shift(-2)  # sometimes there are combos of punctuation
            temp_df = temp_df[~temp_df.words.isin(self.list_puncs)]  # remove "words" that are punctuation
            temp_df.loc[~temp_df['puncs_1_next'].isin(self.list_puncs),
                        'puncs_1_next'] = self.spacekey  # if next word is not puncuation, put a space
            temp_df.loc[~temp_df['puncs_2_next'].isin(self.list_puncs),
                        'puncs_2_next'] = self.spacekey  # if next next word is not puncuation, put a space
            temp_df["punc_next"] = temp_df.apply(
                lambda row: row.puncs_1_next +
                            (row.puncs_2_next if (row.puncs_2_next in self.list_puncs and
                                                  row.puncs_1_next in self.list_puncs) else ""), axis=1)
            temp_df = temp_df.drop(['puncs_1_next', 'puncs_2_next'], axis=1)
            temp_df = temp_df.reset_index(drop=True)
            # done. append to the full dataset
            df = pd.concat([df, temp_df])
            temp_df = None # erase - clear space
        df.reset_index()

        # split into chunks - can combine sentences
        indices_newsent = [i + 1 for i, x in enumerate(list(df["punc_next"])) if
                           any(j in x for j in [".", "!", "?"])]
        indices_newchunk = indices_newsent[::2]  # every chunk = 2 sentences.
        # some chunks exceed MAX_CHUNK_SIZE
        too_long = [i for i, x in enumerate(list(np.diff(indices_newchunk))) if
                    x > self.MAX_CHUNK_SIZE]  # places where the index is too big
        add_endpts = []
        for i in too_long:
            diff = indices_newchunk[i + 1] - indices_newchunk[i]
            split_into_pieces = diff // self.MAX_CHUNK_SIZE + 1
            size = diff / split_into_pieces
            for j in range(1, split_into_pieces):
                add_endpts.append(int(indices_newchunk[i] + j * size))
        indices_newchunk = indices_newchunk + add_endpts
        df["newid"] = (df.index.isin(indices_newchunk)).cumsum()

        # cleaning
        # deal with numbers
        df.loc[df['words'].str.isdigit(), 'words'] = self.numkey  # all numbers will be read as a number
        ##########
        # deal with rare words
        word_counts = nltk.FreqDist(df["words"])
        increasing_count = sorted(word_counts, key=word_counts.get)
        vocab = increasing_count[-self.MAX_VOCAB_SIZE:-1]
        df.loc[~df['words'].isin(vocab), 'words'] = self.UNK  # rare words UNK

        # prepare the words
        vocab.append(self.PAD) # vocab replacing all_words
        vocab.append(self.UNK)
        n_vocab = len(vocab)
        tags = list(set(df["punc_next"].values))
        n_tags = len(tags)

        # process the chunks for training
        getter = ChunkGetter(df)
        chunks = getter.chunks
        word2idx = {w: i for i, w in enumerate(vocab)}  # create dictionaries
        tag2idx = {t: i for i, t in enumerate(tags)}

        #  map the chunks to a sequence of numbers and then pad the sequence.
        X = [[word2idx[w[0]] for w in c] for c in chunks]  # 0 element is the word
        X = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=X, padding="post", value=n_vocab - 1)
        y = [[tag2idx[w[2]] for w in c] for c in chunks]  # 2nd element is the tag
        y = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=y, padding="post", value=tag2idx[self.spacekey])

        # change the y labels to categorical
        y = [to_categorical(i, num_classes=n_tags) for i in y]

        # split into train and test set
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, test_size=0.1)
        self.df = df
        self.vocab = vocab
        self.n_vocab = n_vocab
        self.tags = tags
        self.n_tags = n_tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def my_model(self, units=10, drop=0.1, batch_size=32, epochs=3, validation=0.1, plt_show=True, save_model_params=True):
        input = Input(shape=(self.MAX_CHUNK_SIZE,))
        model = Embedding(input_dim=self.n_vocab, output_dim=self.MAX_CHUNK_SIZE, input_length=self.MAX_CHUNK_SIZE)(input)
        # 50-dim embedding
        model = Dropout(drop)(model)
        model = Bidirectional(LSTM(units=units, return_sequences=True, recurrent_dropout=drop))(model)  # variational biLSTM
        # can use mroe units
        out = TimeDistributed(Dense(self.n_tags, activation="softmax"))(model)  # softmax output layer

        model = Model(input, out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # rmsprop, adam, ?
        history = model.fit(self.X_tr, np.array(self.y_tr), batch_size=batch_size,
                            epochs=epochs, validation_split=validation, verbose=1)
        hist = pd.DataFrame(history.history)
        plt.figure(figsize=(12, 12))
        plt.plot(hist["accuracy"])
        plt.plot(hist["val_accuracy"])
        if plt_show:
            plt.show()
        plt.savefig("accuracy_" + time.strftime("%Y%m%d-%H%M%S") + ".png")

        self.model = model
        if save_model_params:
            self.save_model()

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_" + time.strftime("%Y%m%d-%H%M%S") + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_" + time.strftime("%Y%m%d-%H%M%S") + ".h5")
        print("Saved model to disk")

    def load_model(self, json_name, h5_name):
        json_file = open(json_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(h5_name)
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # score = loaded_model.evaluate(X_te, y_te, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        self.loaded_model = loaded_model

    def model_evaluations(self, this_model, show_eval=True):
        # this_model can be self.model or self.loaded model
        test_eval_true = []
        test_eval_pred = []
        for row in range(self.X_te.shape[0]):
            p = this_model.predict(np.array([self.X_te[row]]))
            p = np.argmax(p, axis=-1)
            for j in range(50):
                if self.vocab[self.X_te[row][j]] != "ENDPAD":
                    pred = p[0][j]
                    true = list(self.y_te[row][j]).index(1)
                    test_eval_true.append(true)
                    test_eval_pred.append(pred)
        self.eval = classification_report(test_eval_true, test_eval_pred)
        if show_eval:
            print(self.eval)

    def predict_new(self, this_model, sent_play="hello"):
        predict_sentence = ""
        words_play = sent_play.split()
        words_play_vocab = [w if w in self.vocab else self.UNK for w in words_play]
        x_play = np.array([[self.word2idx[w] for w in words_play_vocab]])
        # pos_tags = nltk.pos_tag(words)
        x_play = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=x_play, padding="post", value=self.n_vocab - 1)
        prediction = this_model.predict(x_play)
        prediction = np.argmax(prediction, axis=-1)
        for w, pred in zip(words_play, prediction[0]):
            predict_sentence += w + " " + self.tags[pred] + " "
            # clean up
        predict_sentence = predict_sentence.replace(self.spacekey, " ")
        predict_sentence = (" ").join(predict_sentence.split())
        print(predict_sentence)

class ChunkGetter(object):

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



