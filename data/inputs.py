import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, LancasterStemmer, SnowballStemmer
import torch

####################################################################################
######## Padding trigger
####################################################################################
#Set to None if no padding required.
nPad = None


####################################################################################
######## Lemmatization
####################################################################################
lem = SnowballStemmer('english')
nonce = 'kelilili'

class inputs():

    """ In order to add appropriate padding, 0 is reserved for SOS
    and 1 is reserved for EOS, both below and in kgen2.lexemes

    SOS and EOS both still name Kelly :) """

    def __init__(self, lexeme_dictionary, lemmatization=True):
        self.dic=lexeme_dictionary
        self.lemmas = lemmatization
        self.data = None
        super(inputs, self).__init__()

    def bulk_vertical(self, df, columns=['lex']):
        batches=[]
        for sent in df['sent'].unique():
            sent_data=[]
            for col in columns:
                lexemes = []
                if self.lemmas:
                    lexemes=[self.dic[lem.stem(str(w))] for w in df[col].loc[df['sent'].isin([sent])].values]
                else:
                    lexemes = [self.dic[str(w)] for w in df[col].loc[df['sent'].isin([sent])].values]
                sent_data.append(np.array(lexemes).reshape(1, -1))

                batches.append(torch.LongTensor(sent_data).view(-1,1))

        return batches

    def individual_vertical(self, loc, columns=['lex']):
        batch=[]
        for col in columns:
            col_data=[]
            if self.lemmas:
                col_data=[self.dic[lem.stem(str(w))] for w in loc[col].values]
            else:
                col_data = [self.dic[w] for w in loc[col].values]

            batch.append(torch.LongTensor(col_data).view(-1, 1))

        return batch

    def bulk_horizontal(self, df, columns, pad_at=nPad):
        batches=[]
        for sent in df['sent'].unique():
            batch = [df[col].loc[df['sent'].isin([sent])].unique()[0] for col in columns if
                     df[col].loc[df['sent'].isin([sent])].unique()[0] not in [None, np.nan]]
            if self.lemmas:
                batch = [lem.stem(str(w)) for w in batch]
            batch = [self.dic[str(w)] for w in batch]
            if bool(pad_at):
                if len(batch) < pad_at:
                    batch+=[1 for _ in range(pad_at-len(batch))]
            batches.append(torch.LongTensor(batch).view(-1, 1))
        return batches

    def individual_horizontal(self, loc, columns, pad_at=nPad):
        batch = [loc[col].unique()[0] for col in columns if
                 loc[col].unique()[0] not in [None, np.nan]]
        if self.lemmas:
            batch = [lem.stem(str(w)) for w in batch]
        batch = [self.dic[str(w)] for w in batch]
        if bool(pad_at):
            if len(batch) < pad_at:
                batch += [1 for _ in range(pad_at - len(batch))]
        return torch.LongTensor(batch).view(-1, 1)

    def bulk_decoderY(self, df, columns, pad_at=nPad, sparse=True):
        batches = []

        one_hot=[0.0 for _ in range(len(self.dic))]
        for sent in df['sent'].unique():
            batch = [df[col].loc[df['sent'].isin([sent])].unique()[0] for col in columns if
                     df[col].loc[df['sent'].isin([sent])].unique()[0] not in [None, np.nan]]

            if self.lemmas:
                batch = [lem.stem(str(w)) for w in batch]

            batch = [self.dic[str(w)] for w in batch]

            if bool(pad_at):
                if len(batch) < pad_at:
                    batch += [1 for _ in range(pad_at - len(batch))]

            if sparse:
                one_hot_batch=[]
                for i in batch:
                    hot1=list(one_hot)
                    hot1[i] = 1.0
                    one_hot_batch.append(hot1)

                batches.append(torch.tensor(one_hot_batch).view(-1, len(one_hot)))
            else:
                batches.append(torch.tensor(batch).view(-1))

        return batches

    def individual_decoderY(self, loc, columns, pad_at=nPad, sparse=True):
        one_hot = [0.0 for _ in range(len(self.dic))]
        batch = [loc[col].unique()[0] for col in columns if
                 loc[col].unique()[0] not in [None, np.nan]]
        if self.lemmas:
            batch = [lem.stem(str(w)) for w in batch]
        batch = [self.dic[str(w)] for w in batch]

        if bool(pad_at):
            if len(batch) < pad_at:
                batch+=[1 for _ in range(pad_at-len(batch))]

        if sparse:
            one_hot_batch=[]
            for i in batch:
                hot1 = list(one_hot)
                hot1[i] = 1.0
                one_hot_batch.append(hot1)

            return np.array(torch.tensor(one_hot_batch).view(-1, len(one_hot)))
        else:
            return torch.tensor(batch).view(-1)
