## Table of Contents  
- [Overview](#Overview) 
- [Data-Preprocessing](#Data-Preprocessing)  
- [Model](#Model)  
- [Evaluation](#Evaluation)  
- [Future-Considerations](#Future-Considerations)
- [Examples](#Examples)
- [Future-Considerations](#Future-Considerations)
- [Test it out!](#Test it out)



# Overview
Punctuations help to improve comprehension and readability. In this repo, I build a model to automatically restore puncutation marks in unpunctuated text. 

The code in this repo can be used to train a new model. 
This repo also contains code that can be used to experiment with the trained model. This model restores the followingpunctuations: [,.?!] and was trained on the brown corpus and the gutenberg corpus, consisting of a total of ___ words and ____ chunks.

# Data-Preprocessing 
- read in the corpora
- convert sentences to lowercase (no knowledge of casing)
- map every word to the punctuation mark (or space) that follow it 
- remove all punctuation that are not in [,.?!]
- break sentences into chunks of MAX_CHUNK_SIZE (40) and pad if necessary
- replace all numbers with numkey ("9999")
- build vocabulary using the top MAX_VOCAB_SIZE (50000) words. rare words are changed to UNK ("UNK")
- enumerate words and their punctuation tags  
- remove chunks with only one punctuation throughout 

# Model 
- the model is built as a tagging model where each word is tagged with the puncutation that follows it
- Bidirectional LSTM model with 0.1 Dropout for Embedding, Attention, and custom loss function optimized with Adam 
- custom loss function: weighted categorical_crossentropy (weights are inverses of punctuation occurrences, e.g. 1/(#SPACE))
- the example model has 32 units, 10 epochs

# Evaluation
- calculate precision, recall, and f1-score for each punctuation 

# Future-Considerations
- include word embeddings using wordnet or word2vec
- tune parameters of the model to improve diagnostics 
- 


# Examples 

# Test it out

