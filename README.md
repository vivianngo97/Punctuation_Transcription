##### Table of Contents  
[DataPreprocessing](#DataPreprocessing)  
[Emphasis](#emphasis)  
...snip...    
<a name="headers"/>
## Headers


Punctuations help to improve comprehension and readability. In this repo, I build a model to automatically restore puncutation marks in unpunctuated text. 

The code in this repo can be used to train a new model. 
This repo also contains code that can be used to experiment with the trained model. This model restores the followingpunctuations: [,.?!] and was trained on the brown corpus and the gutenberg corpus, consisting of a total of ___ words and ____ chunks.

# DataPreprocessing 
- read in the corpora
- convert sentences to lowercase (no knowledge of casing)
- remove all punctuation that are not in [,.?!]
- break sentences into chunks of MAX_CHUNK_SIZE (40) and pad if necessary
- replace all numbers with numkey ("9999")
- build vocabulary using the top MAX_VOCAB_SIZE (50000) words. rare words are changed to UNK ("UNK")
- enumerate words and their punctuation tags  
- remove chunks with only one punctuation throughout 

# Model 
- the model is built as a tagging model where each word is tagged with the puncutation that follows it
- Bidirectional LSTM model with Attention 
- 32 units, 10 epochs
- custom loss function: weighted categorical_crossentropy (weights are inverses of punctuation occurrences, e.g. 1/(#SPACE))

# Evaluation
- calculate precision, recall, and f1-score for each punctuation 

# Future considerations:
- include word embeddings using wordnet or word2vec
- tune parameters of the model to improve diagnostics 
- 


# Examples 

# Want to test it out?

