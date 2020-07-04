"""
Punctuation Transcription
Author: Vivian Ngo
Date: July 2020
"""

import os
import time  # for time stamp on file names
import pickle
import nltk
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
nltk.download('brown')
nltk.download('gutenberg')
from nltk import corpus
from nltk.corpus import brown
from nltk.corpus import gutenberg
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, InputLayer
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention



class Punc_data(object):
    def __init__(self, corpora=[nltk.corpus.brown, nltk.corpus.gutenberg]):
        self.corpora = corpora
        self.puncs = ',.?!'
        self.list_puncs = list(self.puncs)  # + ["``", "\'\'"]
        self.PAD = "ENDPAD"
        self.UNK = "UNK"
        self.spacekey = "SPACE"
        self.numkey = "9999"
        self.MAX_VOCAB_SIZE = 50000
        self.MAX_CHUNK_SIZE = 40  # max word length per chunk
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
        self.model = None
        self.loaded_model = None
        self.eval = None
        self.eval_df = None
    def preprocess_data(self):
        """
        Preprocess the data so that we can use it for training and testing.
        :return: None
        :rtype: None
        """
        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing \n")
        # gather the data from specified corpora into the required format
        df = pd.DataFrame(columns=['words', 'pos', 'punc_next'])
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
                                                  row.puncs_1_next not in self.list_puncs and
                                                  row.puncs_1_next != self.spacekey and
                                                  row.puncs_1_next != row.puncs_2_next) else ""), axis=1)
            temp_df = temp_df.drop(['puncs_1_next', 'puncs_2_next'], axis=1)
            temp_df = temp_df.reset_index(drop=True)
            # done. append to the full dataset
            df = pd.concat([df, temp_df])
            temp_df = None  # erase - clear space
        df.reset_index()
        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing: splitting into chunks \n")
        # split into chunks - can combine sentences
        indices_newsent = [i + 1 for i, x in enumerate(list(df["punc_next"])) if
                           any(j in x for j in [".", "!", "?"])]
        indices_newchunk = indices_newsent[::2]  # every chunk = 2 sentences.
        # some chunks exceed MAX_CHUNK_SIZE
        too_long = [i for i, x in enumerate(list(np.diff(indices_newchunk))) if
                    x > self.MAX_CHUNK_SIZE]  # places where the index is too big
        print("these chunks are too long: ", str(len(too_long)))
        add_endpts = []
        for i in too_long:
            diff = indices_newchunk[i + 1] - indices_newchunk[i]
            split_into_pieces = diff // self.MAX_CHUNK_SIZE + 1
            size = diff / split_into_pieces
            for j in range(1, split_into_pieces):
                add_endpts.append(int(indices_newchunk[i] + j * size))
        indices_newchunk = indices_newchunk + add_endpts
        df["newid"] = (df.index.isin(indices_newchunk)).cumsum()

        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing: removing chunks with one punctuation \n")

        # remove chunks that have only one punctuation throughout
        temp_df = df.copy()
        counts_unique = temp_df.groupby(["newid"], as_index=True)["punc_next"].nunique().reset_index()
        singles = list(counts_unique[counts_unique["punc_next"] == 1]["newid"])  # has only 1 unique punc
        temp_df = temp_df[~temp_df.newid.isin(singles)]
        temp_df = temp_df.reset_index(drop=True)
        df = temp_df
        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing: cleaning \n")
        # cleaning
        # deal with punctuation
        # df.loc[df['punc_next']==".", 'punc_next'] = "<period>"
        # df.loc[df['punc_next']==",", 'punc_next'] = "<comma>"
        # df.loc[df['punc_next']=="!", 'punc_next'] = "<exclamation>"
        # df.loc[df['punc_next']=="?", 'punc_next'] = "<question_mark>"

        # deal with numbers
        df.loc[df['words'].str.isdigit(), 'words'] = self.numkey  # all numbers will be read as a number
        ##########
        # deal with rare words
        word_counts = nltk.FreqDist(df["words"])
        increasing_count = sorted(word_counts, key=word_counts.get)
        vocab = increasing_count[-self.MAX_VOCAB_SIZE:-1]
        df.loc[~df['words'].isin(vocab), 'words'] = self.UNK  # rare words UNK

        # prepare the words
        vocab.append(self.numkey)
        vocab.append(self.UNK)
        vocab.append(self.PAD)  # vocab replacing all_words
        n_vocab = len(vocab)
        tags = list(set(df["punc_next"].values))
        n_tags = len(tags)

        # process the chunks for training
        getter = ChunkGetter(df)
        chunks = getter.chunks
        word2idx = {w: i for i, w in enumerate(vocab)}  # create dictionaries
        tag2idx = {t: i for i, t in enumerate(tags)}  #

        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing: mapping and padding \n")
        #  map the chunks to a sequence of numbers and then pad the sequence.
        X = [[word2idx[w[0]] for w in c] for c in chunks]  # 0 element is the word
        X = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=X, padding="post", value=n_vocab - 1)

        # one hot encoding for y's
        y = [[tag2idx[w[2]] for w in c] for c in chunks]  # 2nd element is the tag
        y = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=y, padding="post", value=tag2idx[self.spacekey])
        # change the y labels to categorical
        y = [to_categorical(i, num_classes=n_tags) for i in y]
        print(time.strftime("%Y%m%d-%H%M%S") + " pre-processing: train and test set \n")
        # split into train and test set
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, test_size=0.1)
        self.df = df
        self.vocab = vocab
        self.n_vocab = n_vocab
        self.tags = tags
        self.n_tags = n_tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def build_model(self, units=32, drop=0.1, batch_size=32, epochs=10, validation=0.1, plt_show=True,
                    save_model_params=True):
        """
        Build the RNN.
        :param units: Units in LSTM
        :type units: int
        :param drop: Dropout
        :type drop: float
        :param batch_size: Batch size
        :type batch_size: int
        :param epochs: Number of epochs
        :type epochs: int
        :param validation: Proportion of validation set
        :type validation: float
        :param plt_show: Whether we want to show the accuracy plot or not
        :type plt_show: bool
        :param save_model_params:
        :type save_model_params:
        :return: None
        :rtype: None
        """
        # imbalanced classes
        y_ints = [y.argmax() for y in self.y_tr]
        self.class_weights = class_weight.compute_class_weight('balanced',
                                                               np.unique(y_ints),
                                                               y_ints)
        self.class_weights = dict(enumerate(self.class_weights))

        # y_tr2 = [[y.argmax() for y in y_train] for y_train in self.y_tr]]
        # sample_weights = class_weight.compute_sample_weight('balanced', y)
        print(time.strftime("%Y%m%d-%H%M%S") + " building model \n")

        # NEW THINGS FROM HERE
        model = Sequential()
        model.add(InputLayer(input_shape=(self.MAX_CHUNK_SIZE,)))
        model.add(Embedding(input_dim=self.n_vocab, output_dim=self.MAX_CHUNK_SIZE))
        model.add(Dropout(drop))
        model.add(Bidirectional(LSTM(units=units, return_sequences=True)))
        model.add(Dropout(drop))
        model.add(Bidirectional(LSTM(units=units, return_sequences=True)))
        model.add(SeqSelfAttention())
        model.add(TimeDistributed(Dense(self.n_tags)))
        model.add(Activation('softmax'))

        model.compile(loss=weighted_ce,
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])

        model.summary()

        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        print(time.strftime("%Y%m%d-%H%M%S") + " building model: fitting model")
        history = model.fit(self.X_tr, np.array(self.y_tr),
                            batch_size=batch_size,
                            epochs=epochs, validation_split=validation,
                            verbose=1)  # ,
        # class_weight=self.class_weights)
        hist = pd.DataFrame(history.history)
        plt.figure(figsize=(12, 12))
        plt.plot(hist["accuracy"])
        plt.plot(hist["val_accuracy"])
        print(time.strftime("%Y%m%d-%H%M%S") + " building model: accuracy plots")
        if plt_show:
            plt.show()
        plt.savefig("accuracy_" + time.strftime("%Y%m%d-%H%M%S") + ".png")

        self.model = model
        if save_model_params:
            self.save_model()

    def save_model(self, this_model):
        """
        After configuring a model, you can choose to save the model, model weights, and related attributes.
        This function attempts to save them all.
        :param this_model: A model that is to be saved
        :type this_model: keras.engine.sequential.Sequential
        :return: None
        :rtype: None
        """
        # serialize model to JSON
        try:
            model_json = this_model.to_json() # this model can be self.loaded model or self.model
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            with open(timestamp + "_"+ "model" + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            this_model.save_weights(timestamp + "_" + "model" + ".h5")
        except:
            print ("No model attribute")

        try:
            pickle_out = open(timestamp + "_" + "vocab.pickle", "wb")
            pickle.dump(self.vocab, pickle_out)
            pickle_out.close()

            pickle_out = open(timestamp + "_" + "tags.pickle", "wb")
            pickle.dump(self.tags, pickle_out)
            pickle_out.close()

            pickle_out = open(timestamp + "_" + "word2idx.pickle", "wb")
            pickle.dump(self.word2idx, pickle_out)
            pickle_out.close()

            pickle_out = open(timestamp + "_" + "X_te.pickle", "wb")
            pickle.dump(self.X_te, pickle_out)
            pickle_out.close()

            pickle_out = open(timestamp + "_" + "y_te.pickle", "wb")
            pickle.dump(self.y_te, pickle_out)
            pickle_out.close()

            pickle_out = open(timestamp + "_" + "eval.pickle", "wb")
            pickle.dump(self.eval, pickle_out)
            pickle_out.close()
        except:
            print("Object attributes missing")

    def load_model(self, files_directory):
        """
        Loads a model based on the relevant files from the files directory
        :param files_directory:
        :type files_directory:
        :return:
        :rtype:
        """
        # load the relevant parts of the model

        try:
            pickle_in = open(files_directory + "vocab.pickle", "rb")
            self.vocab = pickle.load(pickle_in)
            pickle_in.close()
            self.n_vocab = len(self.vocab)
            print("loaded vocab")
        except:
            print("vocab file not found")

        try:
            pickle_in = open(files_directory + "X_te.pickle", "rb")
            self.X_te = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded X_te")
        except:
            print("X_te file not found")

        try:
            pickle_in = open(files_directory + "y_te.pickle", "rb")
            self.y_te = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded y_te")
        except:
            print("y_te file not found")

        try:
            pickle_in = open(files_directory + "word2idx.pickle", "rb")
            self.word2idx = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded word2idx")
        except:
            print("word2idx file not found")

        try:
            pickle_in = open(files_directory + "tags.pickle", "rb")
            self.tags = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded tags")
        except:
            print("tags file not found")

        try:
            pickle_in = open(files_directory + "eval.pickle", "rb")
            self.eval = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded eval")
        except:
            print("eval file not found")

        try:
            pickle_in = open(files_directory + "eval_df.pickle", "rb")
            self.eval_df = pickle.load(pickle_in)
            pickle_in.close()
            print("loaded eval_df")
        except:
            print("eval_df file not found")

        #try:
        #    pickle_in = open(files_directory + "df.pickle", "rb")
        #    self.df = pickle.load(pickle_in)
        #    pickle_in.close()
        #    print("loaded df")
        #except:
        #    print("df file not found")

        #pickle_in = open(files_directory + "MAX_CHUNK_SIZE.pickle", "rb")
        #self.MAX_CHUNK_SIZE = pickle.load(pickle_in) # already specified when making this object
        #pickle_in.close()
        # pickle_in = open(files_directory + "spacekey.pickle", "rb")
        # self.spacekey = pickle.load(pickle_in)
        # pickle_in.close()
        # pickle_in = open(files_directory + "numkey.pickle", "rb")
        # self.numkey = pickle.load(pickle_in)
        # pickle_in.close()

        # new_obj.loaded_model = tf.keras.models.model_from_json(directory + "model.json") # doesn't work
        loaded_model = model_from_json(open(files_directory + "model.json").read(),
                                       custom_objects={"SeqSelfAttention": SeqSelfAttention})
        print ("loaded model")
        loaded_model.load_weights(os.path.join(os.path.dirname(files_directory + "model.json"), 'model.h5'))
        print("loaded model weights")
        self.loaded_model = loaded_model

    def model_evaluations(self, this_model, show_eval=True):
        print(time.strftime("%Y%m%d-%H%M%S") + " evaluating model \n")
        # just self.model now # old: this_model can be self.model or self.loaded model
        test_eval_true = []
        test_eval_pred = []
        for row in range(self.X_te.shape[0]):
            p = this_model.predict(np.array([self.X_te[row]]))
            p = np.argmax(p, axis=-1)
            for j in range(self.MAX_CHUNK_SIZE):
                if self.vocab[self.X_te[row][j]] != "ENDPAD":
                    pred = p[0][j]
                    true = list(self.y_te[row][j]).index(1)
                    test_eval_true.append(true)
                    test_eval_pred.append(pred)
        self.eval = classification_report(test_eval_true, test_eval_pred, target_names=self.tags,output_dict=True)
        self.eval_df = pd.DataFrame(self.eval).transpose()
        self.eval_df.to_csv('classification_report.csv', index=True)
        if show_eval:
            print(self.eval)

    def predict_new(self, this_model, sent_play="hello"):
        # print(time.strftime("%Y%m%d-%H%M%S") + " making new predictions \n")
        predict_sentence = ""
        sent_play = sent_play.lower()
        words_play = sent_play.split()
        words_play_vocab = [w if w in self.vocab else self.UNK for w in words_play]
        words_play_vocab = [self.numkey if w.isdigit() else w for w in words_play_vocab]
        x_play = np.array([[self.word2idx[w] for w in words_play_vocab]])
        # pos_tags = nltk.pos_tag(words)
        x_play = pad_sequences(maxlen=self.MAX_CHUNK_SIZE, sequences=x_play, padding="post", value=self.n_vocab - 1)
        prediction = this_model.predict(x_play)
        prediction = np.argmax(prediction, axis=-1)
        for w, pred in zip(words_play, prediction[0]):
            predict_sentence += w + " " + self.tags[pred] + " "
            # clean up
        predict_sentence = predict_sentence.replace(self.spacekey, " ")
        # predict_sentence = predict_sentence.replace("pad_punc", " ")
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


def weighted_ce(targets, predictions):

    print(time.strftime("%Y%m%d-%H%M%S") + " weighted cross entropy \n")
    counts = tf.math.reduce_sum(targets, [0, 1])
    print(counts)
    weights = 1 / (counts ** 0.8 + 1)  ###### CAN ALTER THIS
    loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
    print(time.strftime("%Y%m%d-%H%M%S") + " weighted cross entropy: start loop \n")
    custom_loss = 0
    for i in range(5):  # 5 total punctuations
        current_weight = weights[i]
        argmax_targets = tf.argmax(targets, axis=-1)
        current_mask = tf.cast(argmax_targets == i, tf.float32)
        current_loss = current_weight * current_mask * loss
        sum_loss = tf.math.reduce_sum(current_loss)
        print("sum_loss:", custom_loss)
        custom_loss += sum_loss
    print(custom_loss)
    return custom_loss
    # https://stackoverflow.com/questions/43818584/custom-loss-function-in-keras


