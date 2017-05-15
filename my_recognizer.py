import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    test_X_lengths = test_set.get_all_Xlengths()
    
    # loop through all the test words
    for test_word_id in test_X_lengths:
        word_probs = {}
        # get the sequence for the current testword
        X, lengths = test_X_lengths[test_word_id]
        # loop through all word models and calculate the likelihood that the model represents the test word
        for word in models:
            try:
                model = models[word]
                logL = model.score(X, lengths)
                word_probs[word] = logL
            except:
                word_probs[word] = float("-inf")
        probabilities.append(word_probs)
            
    # pick the word model with the highest likelihood for each test word
    for n in range(len(probabilities)):
        guesses.append(max(probabilities[n], key=lambda key: probabilities[n][key]))
        
    return probabilities, guesses
            
