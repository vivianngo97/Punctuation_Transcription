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

The code in this repo can be used to train a new model. __functions.py__ includes the __Punc_data__ object which can be used to build and configure a model, given a list of nltk corpora. 

This repo also contains code that can be used to experiment with a trained example model. This model restores the following punctuations: [,.?!] and was trained on the Brown corpus and the Gutenberg corpus, consisting of a total of 3313299 words and 105804 chunks.

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
- Bidirectional LSTMs with Dropout, Attention, and a custom loss function optimized with Adam 
- Custom weighted categorical crossentropy loss function (weights are inversely related to punctuation occurrences, e.g. SPACE_weight 1/(#SPACE_counts^0.7 + 1)). It is important to use a weighted loss function because of the class imbalance of the punctuation marks (e.g. there are far more spaces than exclamation points).
- The example model has 0.3 dropout, 128 units per layer, and 10 epochs

# Evaluation
After the model is trained, it is then evaluated on a separate testing set based on the following:
- Precision for each punctuation 
- Recall for each punctuation 
- F1-score for each punctuation 

These are the evaluation metrics for the example model: 

|              | precision | recall | f1-score | support     |
| ------------ | --------- | ------ | -------- | ----------- |
| .            | 0.750     | 0.859  | 0.801    | 12156       |
| ?            | 0.254     | 0.618  | 0.360    | 997         |
| ,            | 0.454     | 0.758  | 0.568    | 24248       |
| SPACE        | 0.983     | 0.914  | 0.947    | 292606      |
| !            | 0.153     | 0.379  | 0.218    | 683         |
| accuracy     | 0.899     | 0.899  | 0.899    | 0.899       |
| macro avg    | 0.519     | 0.706  | 0.579    | 330690      |
| weighted avg | 0.932     | 0.899  | 0.911    | 330690      |
|              |

Accuracy is roughly 0.899 percent. However, accuracy is not a sufficient metric to use to evaluate this model because of the class imbalance. In particular, there are a lot more SPACE tags than others. Thus, a model that wrongly tags SPACE everywhere would be incorrect but have high accuracy. In this case, the macro F1-score is a stronger measure of model performance.

As we can see, there is a lack of precision for most punctuation marks, especially those with low support (such as !). For punctuation marks with low precision, they can often be predicted as a tag incorrectly (please see Examples). The corresponding recall values are higher but can still be greatly improved with some enhancements (please see Future Considerations). 

This table of metrics and more evaluations (training and validation accuracy and weighted categorical crossentropy loss) can be found in __model_files/model_evals__.

# Future Considerations
These are some enhancements that could improve the performance of the model.
- Include word embeddings using wordnet or word2vec
- Use POS tag as a feature 
- Map every rare word to a common word in the vocabulary that has the same POS tag (e.g. map all rare proper nouns to "John")
- Tune the weights in the custom loss function to improve the class imbalance issue (1/count makes non-SPACE recall too low, but 1/sqrt(count) causes a lot of SPACE tags)
- Or, optimize macro F1 using a differentiable version of F1 (something like: https://datascience.stackexchange.com/questions/66581/is-it-possible-to-make-f1-score-differentiable-and-use-it-directly-as-a-loss-fun)
- Tune parameters of the model to improve diagnostics 
- Train the model on more corpora and have more layers

# Examples 

<pre><code># my_try is a Punc_data object and my_try.loaded_model is the example model in this repo

>>> my_try.predict_new(my_try.loaded_model, "this is a string of text with no punctuation this is a new sentence")
this is a string of text with no punctuation . this is a new sentence .

>>> my_try.predict_new(my_try.loaded_model, "hello this is a computer program")
hello , this is a computer program .

>>> my_try.predict_new(my_try.loaded_model, "how are you feeling today")
how are you feeling today ?

>>> my_try.predict_new(my_try.loaded_model, "my favorite colors are blue yellow and green")
my favorite colors , are blue , yellow , and green .

>>> my_try.predict_new(my_try.loaded_model, "hello my name is john how are you doing on this fine evening")
hello , my name , is john ! how are you doing on this fine evening ?

>>> my_try.predict_new(my_try.loaded_model, "wow you are amazing")
wow ! you are amazing !

</code></pre>

# Test it out

To test out this punctuation restoration model, follow the steps below. Note that you may be required to install some modules. 

## Via Command Line:
- Clone this repository 
- Navigate to the directory of this repository: Punctuation_Transcription
- Type __python play.py__
- You can now test out the model!
- You will see something like this: 

<pre><code>Puntuation_Transcription> Please type a string and then press ENTER: 

Puntuation_Transcription> You entered: 

Puntuation_Transcription> Let's predict the punctuations:

Puntuation_Transcription> Do you want to play again? Please type yes if you would like to play again: 

Puntuation_Transcription> Have a nice day!
 
</code></pre>

## Via Python IDE
- Clone this repository 
- Run __play.py__
- You can now test out the model! (Similar to Via Command Line)

## Build a model yourself

If you would like to build a model yourself, here is some code to build the example model. The arguments can be configured. Please see __functions.py__ for more details. Please note that the model can take a few hours to build and it is highly recommended to build the model using a GPU. I was able to build the example model in roughly two hours using Google Colab.

<pre><code>import functions
from functions import Punc_data
mypunc = Punc_data([nltk.corpus.brown, nltk.corpus.gutenberg])
mypunc.preprocess_data()
mypunc.build_model(drop=0.3, units=128, epochs=10)
mypunc.model_evaluations(mypunc.model)
mypunc.predict_new(mypunc.model, "this is a string of text with no punctuation this is a new sentence")
</code></pre>


