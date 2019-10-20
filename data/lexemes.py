from random import randint, sample
import pandas as pd
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer, LancasterStemmer, SnowballStemmer

d2vbase = gensim.models.Word2Vec.load('WIKI-d2v-PATH.bin')

nonce = 'kelilili'

####################################################################################
######## Lemmatization
####################################################################################
lem = SnowballStemmer('english')

####################################################################################
####### Embeddings
####################################################################################
class lexemes():

    def __init__(self, with_nonce=False, lemmatization=True):
        self.dic = {'START_with_KELS': 0, 'END_with_Kels': 1}
        self.noncical = with_nonce
        self.lemmatize = lemmatization
        super(lexemes, self).__init__()

    ##If you're starting off with having generated word embeddings,
    # call this guy first.
    def id_dic(self, dfk, input_columns=['tref', 'lex'], fx='Normal'):
        # Creates a single list of lexical items from the summ total of
        # lexical units being passed to the network . . .
        lexemes = []
        for col in input_columns:
            for word in dfk[col].unique():
                if self.lemmatize:
                    lexemes.append(lem.stem(str(word)))
                else:
                    lexemes.append(str(word))

        # this is necessary for conversion of items in batch_sents(),
        # where each word will be replaced with its vector rep in the
        # network. We start by creating an id#-to-word dictionary, and
        # then just flip it to have a word-to id#, where the ID# is the
        # index of the word in the list of vectors.
        ct = len(self.dic)
        for word in set(lexemes):
            if word not in self.dic.keys():  # delete this to go back to old version
                self.dic[word] = ct
                ct += 1

        return self.dic

    def embeds(self, word2id, mod=d2vbase, dimensions=300):
        vec_vocab = [0 for word in range(len(word2id))]
        # Here we take the empty list vec_vocab, and fill it with vectors
        # for our document vocab. Also collects info about KeyErrors...
        errors = []
        for word, idx in word2id.items():
            try:
                vec_vocab[word2id[word]] = mod.wv[word]
            except KeyError:
                vec_vocab[word2id[word]] = np.random.rand(dimensions)
                if word != str(nonce):
                    errors.append(word)

        # Finally, we return the list of vectors (a list of np.arrays) and
        # our conversion dictionary, mapping words to vec_vocab indeces,
        # which are then used in the LSTM batches.
        print(len(set(errors)), ' vocabulary items not translated into vecs.')
        return np.array(vec_vocab), errors

    def embeds_init_1s(self, word2id, mod, dimensions=300):
        vec_vocab = [0 for word in range(len(word2id))]
        # Here we take the empty list vec_vocab, and fill it with vectors
        # for our document vocab. Also collects info about KeyErrors...
        errors = []
        for word, idx in word2id.items():
            try:
                vec_vocab[word2id[word]] = mod.wv[word]
            except KeyError:
                vec_vocab[word2id[word]] = np.array([1 for _ in range(dimensions)])
                if word != str(nonce):
                    errors.append(word)

        # Finally, we return the list of vectors (a list of np.arrays) and
        # our conversion dictionary, mapping words to vec_vocab indeces,
        # which are then used in the LSTM batches.
        print(len(set(errors)), ' vocabulary items not translated into vecs.')
        return np.array(vec_vocab), errors

    def dic_persist(self, dic, filename='session-dic.csv'):
        df = pd.DataFrame(np.array(list(zip(list(dic.keys()), list(dic.values())))
                                   ).reshape(-1, 2), columns=['keys', 'values']
                          )
        df.to_csv(path_or_buf='kgen/1-data/_sessions/' + filename, index=False, encoding='utf-8')

    def load_from_file(self, filename):
        df = pd.read_csv('kgen/1-data/_sessions/' + filename)
        for row in df.values.tolist():
            self.dic[row[0]] = row[1]
        return self.dic
