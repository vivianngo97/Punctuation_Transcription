if __name__ == "__main__":
    import os
    import functions
    from functions import Punc_data
    my_dir = os.getcwd() + "/model_files/"
    my_try = Punc_data()
    my_try.load_model(my_dir)
    play = True
    while play:
        user_input_sentence = input("\nPlease type an unpunctuated string and then press ENTER: ")
        print(f'\nYou entered: {user_input_sentence}')
        print("\nLet's predict the punctuations: \n")
        my_try.predict_new(my_try.loaded_model, sent_play=user_input_sentence)
        again = input("\nDo you want to play again? Please type yes if you would like to play again: ")
        if again != "yes":
            play = False
    print ("\nHave a nice day!")

# my_try.model_evaluations(this_model = my_try.loaded_model)
