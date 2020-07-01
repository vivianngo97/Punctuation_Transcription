import random
import os
import sys
import re
from io import open
import operator

try:
    import cPickle
except:
    import _pickle as cPickle

DATA_PATH = "../data"
TRAIN_FILE = os.path.join(DATA_PATH, "train")
DEV_FILE = os.path.join(DATA_PATH, "dev")
TEST_FILE = os.path.join(DATA_PATH, "test")
WORD_VOCAB_FILE = os.path.join(DATA_PATH, "vocabulary")

END = "</S>"
UNK = "<UNK>"
NUM = "<NUM>"
SPACE = "_SPACE"
MIN_WORD_COUNT_VOCAB = 5
MAX_VOCAB_SIZE = 10000

PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON",
                          "-DASH", "'APOSTROPHE", "\"QUOTATION"]

# load and dump functions with pickle that use less RAM
def dump(d, path):
    with open(path, 'w') as f:
        for s in d:
            f.write("%s\n" % repr(s))


def load(path):
    d = []
    with open(path, 'r') as f:
        for l in f:
            d.append(eval(l))
    return d


def add_words(word_counts, line):
    """
    Add words to the word count
    :param word_counts:
    :type word_counts: dict
    :param line: new line to get words from
    :type line: str
    :return: None
    :rtype: None
    """
    for w in re.split(pattern='[ ,.?!:;\-\'\"]', string=line):
        if w not in PUNCTUATION_VOCABULARY and w != "":
            word_counts[w] = word_counts.get(w, 0) + 1


def build_vocab(word_counts):
    """
    build the vocabulary and only keep the MAX_VOCAB_SIZE most frequent words
    :param word_counts:
    :type word_counts:
    :return:
    :rtype:
    """
    return [w[0] for w in sorted(word_counts.items(), key=operator.itemgetter(1))
            if w[1] >= MIN_WORD_COUNT_VOCAB and w[0] != UNK][:MAX_VOCAB_SIZE]


def write_vocab(vocabulary, file_name):
    """
    write to the vocabulary
    :param vocabulary:
    :type vocabulary:
    :param file_name:
    :type file_name:
    :return:
    :rtype:
    """
    if END not in vocabulary:
        vocabulary.append(END)
    if UNK not in vocabulary:
        vocabulary.append(UNK)

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(vocabulary))


def iterable_to_dict(arr):
    return dict((x.strip(), i) for (i, x) in enumerate(arr))


def read_vocabulary(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return iterable_to_dict(f.readlines())


def write_processed_data(input_files, output_file):
    """
    write the data into a file.
    One sequence of words, one corresponding sequence of punctuation
    No punctuation
    :param input_file:
    :type input_file:
    :param output_file:
    :type output_file:
    :return:
    :rtype:
    """
    data = []

    word_vocab = read_vocabulary(WORD_VOCAB_FILE)
    punc_vocab = iterable_to_dict(PUNCTUATION_VOCABULARY)

    num_total = 0
    num_unks = 0

    current_words = []
    current_punctuations = []

    last_eos_index = 0  # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_token_was_punctuation = True  # skipt first token if it's punctuation

    skip_until_eos = False  # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence

    for file in input_files:
        with open(file, 'r', encoding='utf-8') as text:
            for line in text:




