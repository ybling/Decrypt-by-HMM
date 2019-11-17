#!/usr/bin/env python
# coding: utf-8

# In[7]:
import argparse
parser = argparse.ArgumentParser(description='')


import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk import LaplaceProbDist
from nltk.probability import (
    FreqDist,
    ConditionalFreqDist,
    ConditionalProbDist,
    DictionaryProbDist,
    DictionaryConditionalProbDist,
    LidstoneProbDist,
    MutableProbDist,
    MLEProbDist,
    RandomProbDist,
)
import os

# In[8]:
# parser.add_argument("cipher_folder")


# args = parser.parse_args()


# fileCipherTrain1 = os.path.join(args.cipher_folder, "train_cipher.txt")
# filePlainTrain1 = os.path.join(args.cipher_folder, "train_plain.txt")
# fileCipherTest1 = os.path.join(args.cipher_folder, "test_cipher.txt")
# filePlainTest1 = os.path.join(args.cipher_folder, "test_plain.txt")

# fileCipherTrain2 = r"E:\data\cipher2\train_cipher.txt"
# filePlainTrain2 = r"E:\data\cipher2\train_plain.txt"
# fileCipherTest2 = r"E:\data\cipher2\test_cipher.txt"
# filePlainTest2 = r"E:\data\cipher2\test_plain.txt"

# fileCipherTrain3 = r"E:\data\cipher3\train_cipher.txt"
# filePlainTrain3 = r"E:\data\cipher3\train_plain.txt"
# fileCipherTest3 = r"E:\data\cipher3\test_cipher.txt"
# filePlainTest3 = r"E:\data\cipher3\test_plain.txt"


# data preprocessing

# In[9]:


def trainpreprocessing(file1, file2):
    train_data = []
    with open(file1,"r") as fc, open(file2,"r") as fp:
        temp = []
        linecs = fc.readlines()
        lineps = fp.readlines()
        
    for j in range(len(linecs)):
        linec = linecs[j].strip('').strip('\n')
        linep = lineps[j].strip('').strip('\n')
        for i in range(len(linec)):
            t = (linec[i], linep[i])
            temp.append(t)
        train_data.append(temp)
    return train_data

def testpreprocessing(file1, file2):
    test_cipher = []
    test_plain = []
    with open(file1,"r") as fc, open(file2,"r") as fp:
        linecs = fc.readlines()
        lineps = fp.readlines()
        
    for j in range(len(linecs)):
        linec = linecs[j].strip('').strip('\n')
        linep = lineps[j].strip('').strip('\n')
        for i in range(len(linec)):
            t = (linec[i], linep[i])
            test_cipher.append(linec[i])
            test_plain.append(linep[i])
    return test_cipher, test_plain


# In[10]:


supplement = r"supplement_data.txt"
def addData():
    chats = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ',',','.']
    data = []
    with open(supplement,"r") as fr:
        linecs = fr.readlines()
        for j in range(len(linecs)):
            linec = linecs[j].strip('').strip('\n').strip().lower()
            temp = []
            for i in range(len(linec)-1):
                a = linec[i]
                b = linec[i+1]
                if a not in chats or b not in chats:
                    i += 1
                    continue
                i += 1
                t = (a, b)
                temp.append(t)
            data.append(temp)
            
    return data
data = addData()


# evaluation

# In[11]:


def eval(predc, corr):
    correct_num = 0
    total_num = 0
    prediction = "".join(predc)
    print(prediction)
    for i in range(len(predc)):
        if predc[i] == corr[i]:
            correct_num += 1
        total_num += 1
    acc = correct_num/total_num
    print(acc)


# In[12]:


# train_data1 = trainpreprocessing(fileCipherTrain1, filePlainTrain1)
# # train_data2 = trainpreprocessing(fileCipherTrain2, filePlainTrain2)
# # train_data3 = trainpreprocessing(fileCipherTrain3, filePlainTrain3)

# test_cipher1, test_plain1 = testpreprocessing(fileCipherTest1, filePlainTest1)
# # test_cipher2, test_plain2 = testpreprocessing(fileCipherTest2, filePlainTest2)
# # test_cipher3, test_plain3  = testpreprocessing(fileCipherTest3, filePlainTest3)


# Standard HMM

# In[13]:

def standard_HMM():
    print("==================Standard HMM  ========================")
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger1 = trainer.train_supervised(train_data1)
    predc1 = tagger1.best_path_simple(test_cipher1)
    eval(predc1, test_plain1)


    # # In[14]:


    # print("========================Standard HMM =========================")
    # trainer = hmm.HiddenMarkovModelTrainer()
    # tagger2 = trainer.train_supervised(train_data2)
    # predc2 = tagger2.best_path_simple(test_cipher2)
    # eval(predc2, test_plain2)


    # # In[15]:


    # print("===============================Standard HMM ===============================")
    # trainer = hmm.HiddenMarkovModelTrainer()
    # tagger3 = trainer.train_supervised(train_data3)
    # predc3 = tagger3.best_path_simple(test_cipher3)
    # eval(predc3, test_plain3)


# Laplace smoothing

# In[16]:

def Laplase_smoothing():

    print("===============================Laplace smoothing ===============================")
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger1 = trainer.train_supervised(train_data1,estimator=LaplaceProbDist)
    predc1 = tagger1.best_path_simple(test_cipher1)
    eval(predc1, test_plain1)


    # In[17]:


    # print("===============================Laplace smoothing for cipher2===============================")
    # trainer = hmm.HiddenMarkovModelTrainer()
    # tagger2 = trainer.train_supervised(train_data2,estimator=LaplaceProbDist)
    # predc2 = tagger2.best_path_simple(test_cipher2)
    # eval(predc2, test_plain2) 


    # # In[18]:


    # print("===============================Laplace smoothing for cipher3===============================")
    # trainer = hmm.HiddenMarkovModelTrainer()
    # tagger3 = trainer.train_supervised(train_data3,estimator=LaplaceProbDist)
    # predc3 = tagger3.best_path_simple(test_cipher3)
    # eval(predc3, test_plain3)


# Add extend data for Improved plaintext modelling

# In[20]:


#define subclass from hmm.HiddenMarkovModelTrainer
class MyHmmTrainer(hmm.HiddenMarkovModelTrainer):

    def train_supervised(self, labelled_sequences,data, estimator=None):
        _TEXT = 0  # index of text in a tuple
        _TAG = 1  # index of tag in a tuple
        # default to the MLE estimate
        if estimator is None:
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurrences of starting states, transitions out of each state
        # and output symbols observed in each state
        known_symbols = set(self._symbols)
        known_states = set(self._states)

        starting = FreqDist()
        transitions = ConditionalFreqDist()
        outputs = ConditionalFreqDist()
        for sequence in labelled_sequences:
            lasts = None
            for token in sequence:
                state = token[_TAG]
                symbol = token[_TEXT]
                if lasts is None:
                    starting[state] += 1
                else:
                    transitions[lasts][state] += 1
                outputs[state][symbol] += 1
                lasts = state
                
        for sequence in data:
            lasts = None
            for token in sequence:
                state = token[_TAG]
                symbol = token[_TEXT]
                if lasts is None:
                    starting[state] += 1
                else:
                    transitions[lasts][state] += 1
                #outputs[state][symbol] += 1
                lasts = state

                # update the state and symbol lists
                if state not in known_states:
                    self._states.append(state)
                    known_states.add(state)

                if symbol not in known_symbols:
                    self._symbols.append(symbol)
                    known_symbols.add(symbol)

        # create probability distributions (with smoothing)
        N = len(self._states)
        pi = estimator(starting, N)
        A = ConditionalProbDist(transitions, estimator, N)
        B = ConditionalProbDist(outputs, estimator, len(self._symbols))

        return hmm.HiddenMarkovModelTagger(self._symbols, self._states, A, B, pi)


# In[21]:

def improved_plaintext_modelling():
    print("========================Improved plaintext modelling ===========================")
    t = MyHmmTrainer()
    tagger1 = t.train_supervised(train_data1,data)
    predc1 = tagger1.best_path_simple(test_cipher1)
    eval(predc1, test_plain1)


    # # In[23]:


    # print("========================Improved plaintext modelling for cipher2===========================")
    # t = MyHmmTrainer()
    # tagger2 = t.train_supervised(train_data2,data)
    # predc2 = tagger2.best_path_simple(test_cipher2)
    # eval(predc2, test_plain2)


    # # In[24]:


    # print("========================Improved plaintext modelling for cipher3===========================")
    # t = MyHmmTrainer()
    # tagger3 = t.train_supervised(train_data3,data)
    # predc3 = tagger3.best_path_simple(test_cipher3)
    # eval(predc3, test_plain3)


# parser.add_argument('-lm', dest="func", action="store_const", const=improved_plaintext_modelling, default=standard_HMM,   help='')
# parser.add_argument('-laplace', dest="func", action="store_const", const=Laplase_smoothing,   help='')
# args = parser.parse_args()
# args.func()
parser.add_argument("cipher_folder")
parser.add_argument('-lm', dest="func", action="store_const", const=improved_plaintext_modelling, default=standard_HMM, help='')
parser.add_argument('-laplace', dest="func", action="store_const", const=Laplase_smoothing,   help='')
args = parser.parse_args()

if __name__ == "__main__":

    fileCipherTrain1 = os.path.join(args.cipher_folder, "train_cipher.txt")
    filePlainTrain1 = os.path.join(args.cipher_folder, "train_plain.txt")
    fileCipherTest1 = os.path.join(args.cipher_folder, "test_cipher.txt")
    filePlainTest1 = os.path.join(args.cipher_folder, "test_plain.txt")
    train_data1 = trainpreprocessing(fileCipherTrain1, filePlainTrain1)
    test_cipher1, test_plain1 = testpreprocessing(fileCipherTest1, filePlainTest1)
    args.func()
