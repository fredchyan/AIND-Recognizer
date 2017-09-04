import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        res = (None, None) # (BIC Score, Model)
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                curr_model = self.base_model(num_states)
                if curr_model is not None:
                    curr_logL = curr_model.score(self.X, self.lengths)
                    num_param_transition_probs = num_states * (num_states - 1)
                    num_param_starting_probs = num_states - 1
                    num_param_means = num_states * len(self.X[0]) # len(self.X[0]) is number of features
                    num_param_covar = num_states * len(self.X[0])
                    p = num_param_transition_probs + num_param_starting_probs + num_param_means + num_param_covar
                    curr_BIC = -2 * curr_logL + p * math.log(len(self.lengths))
                    if res[0] is None or curr_BIC < res[0]:
                        res = (curr_BIC, curr_model)
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
        return res[1]

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        best_DIC = None
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                curr_model = self.base_model(num_states)
                sum_all_but_i = 0
                m_minus_1 = 0
                if curr_model is not None:
                    for word, sequence in self.words.items():
                        if word == self.this_word:
                            continue
                        word_X, word_Xlengths = self.hwords[word]
                        try:
                            word_logL = curr_model.score(word_X, word_Xlengths)
                        except:
                            continue
                        sum_all_but_i += word_logL
                        m_minus_1 += 1
                    curr_logL = curr_model.score(self.X, self.lengths)
                    curr_DIC = curr_logL - (1/m_minus_1) * sum_all_but_i
                    if best_model == None or curr_DIC > best_DIC:
                        best_DIC = curr_DIC
                        best_model = curr_model 
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        if len(self.sequences) <= 2:
            return self.base_model(self.n_constant)
        best_num_components = None
        max_k_logL = None
        split_method = KFold()
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            k_logL = None
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                train_X = []
                train_lengths = []
                for i in cv_train_idx:
                    train_X += self.sequences[i]
                    train_lengths.append(len(self.sequences[i]))
                test_X = []
                test_lengths = []
                for i in cv_test_idx:
                    test_X += self.sequences[i]
                    test_lengths.append(len(self.sequences[i]))
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    logL = hmm_model.score(test_X, test_lengths)
                    if k_logL is None:
                        k_logL = logL
                    else:
                        k_logL += logL
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
            if max_k_logL is None or k_logL is not None and k_logL > max_k_logL:
                max_k_logL = k_logL
                best_num_components = num_states
        return self.base_model(best_num_components)
