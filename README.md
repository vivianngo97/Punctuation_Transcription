# Table of Contents  
- [Overview](#Overview) 
- [Data Preprocessing](#Data-Preprocessing)  
- [Model](#Model)  
- [Evaluation](#Evaluation)  
- [Future Considerations](#Future-Considerations)
- [Examples](#Examples)
- [Test it out!](#Test-it-out)



# Overview
Punctuations help to improve comprehension and readability. In this repo, I build a model to automatically restore puncutation marks in unpunctuated text. 

The code in this repo can be used to train a new model. __functions.py__ includes the Punc_data object which can be used to build and configure a model, given a list of nltk corpora. 

This repo also contains code that can be used to experiment with a trained example model. This model restores the following punctuations: [,.?!] and was trained on the Brown corpus and the Gutenberg corpus, consisting of a total of ___ words and ____ chunks.

# Data Preprocessing 
These are the data preprocessing steps: 
- read in the specified corpora (Brown and Gutenberg for the example model)
- convert sentences to lowercase (no knowledge of casing)
- map every word to the punctuation mark (or space) that follow it 
- remove all punctuation that are not in [,.?!]
- break sentences into chunks of MAX_CHUNK_SIZE (40) and pad if necessary
- replace all numbers with numkey ("9999")
- build vocabulary using the top MAX_VOCAB_SIZE (50000) common words. Other worsd are changed to UNK ("UNK")
- enumerate words and their punctuation tags  
- remove chunks with only one punctuation throughout 

# Model 
I have chosen to frame punctuation restoration as a sequence tagging problem where each word is tagged with the punctuation that follows it. The model is a bidirectional recurrent neural network model with the following specifications:
- Bidirectional LSTM model with 0.1 Dropout for Embedding, Attention, and custom loss function optimized with Adam 
- custom loss function: weighted categorical crossentropy (weights are inverses of punctuation occurrences, e.g. 1/(#SPACE))
- the example model has 32 units, 10 epochs

# Evaluation
After the model is trained, it is then evaluated on a separate testing set based on the following:
- Precision for each punctuation 
- Recall for each punctuation 
- F1-score for each punctuation 

# Future Considerations
These are some enhancements that could improve the performance of the model.
- include word embeddings using wordnet or word2vec
- tune parameters of the model to improve diagnostics 
- use POS tag as a feature 
- map rare words to a common word that has the same POS tag (e.g. map all rare proper nouns to "John")

# Examples 

# Test it out

