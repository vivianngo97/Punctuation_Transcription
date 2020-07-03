# Table of Contents  
- [Overview](#Overview) 
- [Data Preprocessing](#Data-Preprocessing)  
- [Model](#Model)  
- [Evaluation](#Evaluation)  
- [Future Considerations](#Future-Considerations)
- [Test it out!](#Test-it-out)


# Overview
Punctuations help to improve comprehension and readability. In this repo, I build a model to automatically restore puncutation marks in unpunctuated text. 

The code in this repo can be used to train a new model. __functions.py__ includes the Punc_data object which can be used to build and configure a model, given a list of nltk corpora. 

This repo also contains code that can be used to experiment with a trained example model. This model restores the following punctuations: [,.?!] and was trained on the Brown corpus and the Gutenberg corpus, consisting of a total of _____ words and ____ chunks.

# Data Preprocessing 
These are the data preprocessing steps: 
- Read in the specified corpora (Brown and Gutenberg for the example model)
- Convert sentences to lowercase (no knowledge of casing)
- Map every word to the punctuation mark (or space) that follow it 
- Remove all punctuation that are not in [,.?!]
- Break sentences into chunks of MAX_CHUNK_SIZE (40) and pad if necessary
- Replace all numbers with numkey ("9999")
- Build vocabulary using the MAX_VOCAB_SIZE (50000) top common words. Other worsd are changed to UNK ("UNK")
- Enumerate words and their punctuation tags  
- Remove chunks with only one punctuation throughout 

# Model 
I have chosen to frame punctuation restoration as a sequence tagging problem where each word is tagged with the punctuation that follows it. The model is a bidirectional recurrent neural network model with the following specifications:
- Bidirectional LSTM model with 0.1 Dropout for Embedding, Attention, and custom loss function optimized with Adam 
- Custom weighted categorical crossentropy loss function (weights are inverses of punctuation occurrences, e.g. 1/(#SPACE^0.75 + 1)). It is important to use a weighted loss function because of the class imbalance of the punctuation marks (e.g. there are far more spaces than exclamation points).
- The example model has 64 units, 10 epochs

# Evaluation
After the model is trained, it is then evaluated on a separate testing set based on the following:
- Precision for each punctuation 
- Recall for each punctuation 
- F1-score for each punctuation 

These are the evaluation metrics for the example model: 

|              | precision | recall | f1-score | support     |
| ------------ | --------- | ------ | -------- | ----------- |
| !            | 0.115     | 0.307  | 0.167    | 625         |
| SPACE        | 0.980     | 0.903  | 0.940    | 292840      |
| .            | 0.732     | 0.817  | 0.772    | 12301       |
| ,            | 0.406     | 0.728  | 0.521    | 24200       |
| ?            | 0.223     | 0.528  | 0.314    | 1018        |
| accuracy     | 0.885     | 0.885  | 0.885    | 0.884517076 |
| macro avg    | 0.491     | 0.657  | 0.543    | 330984      |
| weighted avg | 0.925     | 0.885  | 0.900    | 330984      |

Accuracy is roughly 0.885 percent. However, accuracy is not a sufficient metric to use to evaluate this model because of the class imbalance. In particular, there are a lot more spaces than other classes. Thus, a model that wrongly tags spaces everywhere would be incorrect but have high accuracy. In this case, the macro F1-score is a stronger proxy od model performance.

As we can see, there is a lack of precision for most punctuation marks. Recall is higher but is still not as high as it can possibly be, with some more enhancements (see below). 

# Future Considerations
These are some enhancements that could improve the performance of the model.
- Include word embeddings using wordnet or word2vec
- Use POS tag as a feature 
- Map every rare word to a common word in the vocabulary that has the same POS tag (e.g. map all rare proper nouns to "John")
- Tune the weights in the custom loss function to improve the class imbalance issue (1/count makes non-SPACE recall too low, but 1/sqrt(count) causes a lot of SPACE tags)
- Or, optimize macro F1 using a differentiable version of F1 (something like: https://datascience.stackexchange.com/questions/66581/is-it-possible-to-make-f1-score-differentiable-and-use-it-directly-as-a-loss-fun)
- Tune parameters of the model to improve diagnostics 
- Train the model on more corpora and have more layers

# Test it out

To test out this punctuation restoration model, follow the steps below. Note that you may be required to install some modules. 

## Via Command Line:
- Clone this repository 
- Navigate to the directory of this repository: Punctuation_Transcription
- Type python play.py
- You can now test out the model!
- You will see something like this: 

<pre><code>
Puntuation_Transcription> Please type a string and then press ENTER: 

Puntuation_Transcription> You entered: 

Puntuation_Transcription> Let's predict the punctuations:

Puntuation_Transcription> Do you want to play again? Please type yes if you would like to play again: 

Puntuation_Transcription> Have a nice day!
 
</code></pre>

## Via Python IDE
- Clone this repository 
- Run play.py
- You can now test out the model!



