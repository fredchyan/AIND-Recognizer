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
    # implement the recognizer
    # For every word_id in test_set, score it against each trained word model
    # , store the log likelihood and also report the best word guess
    for word_id, Xlengths in test_set.get_all_Xlengths().items():
      curr_dict = dict()
      best_guess = None
      best_ll = None
      for possible_word, possible_word_model in models.items():
        try:
          curr_ll = possible_word_model.score(Xlengths[0],Xlengths[1])
        except:
          continue
        curr_dict[possible_word] = curr_ll
        if not best_guess or curr_ll > best_ll:
          best_guess = possible_word
          best_ll = curr_ll
      probabilities.append(curr_dict)
      guesses.append(best_guess)
    return probabilities, guesses
