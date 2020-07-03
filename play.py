import os
import functions
from functions import Punc_data
import pickle
from keras.models import model_from_json

def play_model(directory, user_input):
    # load the relevant parts of the model
    new_obj = Punc_data()  # initialize

    pickle_in = open(directory + "vocab.pickle", "rb")
    new_obj.vocab = pickle.load(pickle_in)
    pickle_in.close()

    #pickle_in = open(directory + "tags.pickle", "rb")
    #new_obj.tags = pickle.load(pickle_in)
    #pickle_in.close()

    new_obj.tags = ['.', ',', 'SPACE', '!', '?']

    pickle_in = open(directory + "word2idx.pickle", "rb")
    new_obj.word2idx = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(directory + "MAX_CHUNK_SIZE.pickle", "rb")
    new_obj.MAX_CHUNK_SIZE = pickle.load(pickle_in)
    pickle_in.close()

    new_obj.n_vocab = len(new_obj.vocab)
    new_obj.spacekey = "SPACE"
    # new_obj.loaded_model = tf.keras.models.model_from_json(directory + "model.json") # doesn't work

    loaded_model = model_from_json(open(directory + "model.json").read())
    loaded_model.load_weights(os.path.join(os.path.dirname(directory + "model.json"), 'model.h5'))

    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(directory + "model.h5")

    new_obj.loaded_model = loaded_model
    new_obj.predict_new(this_model=new_obj.loaded_model, sent_play="hello my name is vivian")



if __name__ == "__main__":
    my_dir = os.getcwd() + "/model_files/"
    play_model(directory=my_dir, user_input="please write your sentence here"):
