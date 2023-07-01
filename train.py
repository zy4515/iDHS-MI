#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import sys, argparse
import os
os.chdir('/data/home/zongyu/iDHS_MI')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#import pandas as pd
from keras.models import Sequential
from turtle import xcor
from tensorflow import keras
from sklearn import metrics
from keras.metrics import binary_accuracy
import warnings
from keras.optimizers import SGD, Adam
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
warnings.filterwarnings("ignore")
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Conv1D, Input, MaxPooling1D, Flatten, LeakyReLU, AveragePooling1D, concatenate, \
    Multiply, Bidirectional,Concatenate
from keras import regularizers
from keras.models import Input, Model
from keras.layers.core import Permute, Reshape,  Lambda, K, RepeatVector
from sklearn.metrics import confusion_matrix
from metrics_plot import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
import _pickle as cPickle
from keras.preprocessing import sequence
import collections
from sklearn.model_selection import StratifiedKFold
#from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention
from self_attention import SeqSelfAttention
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2


def Trans(seq):
    X = []
    dic = {'A':1,'C':2,'G':6,'T':17}
    for i in range(len(seq)):
        X.append(dic.get(seq[i]))
    return X

def createTrainData(str1):
    sequence_num = []
    label_num = []
    f = open(str1).readlines()
    for i in range(0,len(f)-1,2):
        label = f[i].strip('\n').replace('>','')
        label_num.append(int(label))
        sequence = f[i+1].strip('\n')
        sequence_num.append(sequence)

    return sequence_num,label_num

def CCN(seq):
    count_A = 0
    count_C = 0
    count_G = 0
    count_T = 0
    count_len = 0
    ccn_dict = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0, ], 'T': [0, 0, 1]}
    CCN_matrix = np.zeros((len(seq), 4))
    for l, s in enumerate(seq):
        for i, char in enumerate(s):
            count_len += 1
            if char == 'A':
                count_A +=1
                density = count_A/count_len
                CCN_matrix[l] = (ccn_dict['A']+[density])
            if char == 'C':
                count_C +=1
                density = count_C/count_len
                CCN_matrix[l] = (ccn_dict['C']+[density])
            if char =='G':
                count_G +=1
                density = count_G/count_len
                CCN_matrix[l] = (ccn_dict['G']+[density])
            if char == 'T':
                count_T +=1
                density = count_T/count_len
                CCN_matrix[l] = (ccn_dict['T']+[density])
    return CCN_matrix

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    #print(word_index)
    return word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch1 + ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    #print(word_index)
    return word_index

def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq, kmer, coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq, kmer, tris):
    coden_dict = tris
    freq_dict = frequency(seq, kmer, coden_dict)
    vectors = np.zeros((len(seq), len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i + kmer]]]
        vectors[i][coden_dict[seq[i:i + kmer]]] = 1
    return vectors


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array


def OneHot(seq):
    onehot_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    OneHot_matrix = np.zeros((len(seq), 4))
    for l, s in enumerate(seq):
        for i, char in enumerate(s):
            if char in onehot_dict:
                OneHot_matrix[l] = (onehot_dict[char])
    return OneHot_matrix


def Hod2(sequence):
    tris2 = get_2_trids()
    kmer2 = coden(sequence, 2, tris2)
    return kmer2

def C_MEMBE(OneHot_matrix,k):
    C_MEMBE_matrix = np.zeros((len(OneHot_matrix), 4*k))
    for i in range(0,len(OneHot_matrix)):
        l=i
        for j in range(0,4*k,4):
            C_MEMBE_matrix[i][j:j+4] = OneHot_matrix[l]
            l+=1
            if l>len(OneHot_matrix)-1:
                break
    return C_MEMBE_matrix

    return C_MEMBE_matrix

def C_NCPNF(CCN_matrix,k):
    C_NCPNF_matrix = np.zeros((len(CCN_matrix)-k+1, 4*k))
    for i in range(0,len(CCN_matrix)-k+1):
        l=i
        for j in range(0,4*k,4):
            C_NCPNF_matrix[i][j:j+4] = CCN_matrix[l]
            l+=1

    return C_NCPNF_matrix

def seq_matrix(X_train,labels,k1,k2):
    sequence_num1 = []
    sequence_num2 = []
    sequence_num3 = []
    sequence_num4 = []
    f = X_train
    for i in range(0,len(f)):
        sequence = f[i].strip('\n')
        sequence_num1.append(Trans(sequence)) 
        CCN_matrix = CCN(sequence)
        sequence_num2.append(C_NCPNF(CCN_matrix,k1))
        OneHot_matrix = OneHot(sequence)
        sequence_num3.append(C_MEMBE(OneHot_matrix,k2))
        sequence_num4.append(Hod2(sequence))

    X =sequence_num1
    X1 = [[1] + [w + 3 for w in x] for x in X]   
    X_train1 = X1
    X_train2 = sequence_num2
    X_train3 = sequence_num3
    X_train4 = sequence_num4

    X_train1 = np.array(X_train1)
    X_train2 = np.array(X_train2)
    X_train3 = np.array(X_train3)
    X_train4 = np.array(X_train4)
    y_train = np.array(labels)

    return X_train1, X_train2, X_train3, X_train4,y_train

def main():

    if not os.path.exists(args.output):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.output)
    if not os.path.exists(args.inputfile):
        print("The input data not exist! Error\n")
        sys.exit()

    funciton(args.inputfile, args.output)

def Model1():
    inp1 = Input(shape=(300,))
    x1 = Embedding(23, 128, input_length=300)(inp1)
    x1 = Dropout(0.5)(x1)
    x1 =Conv1D(filters=64, kernel_size=10, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(x1)
    x1 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x1)
    x1 =Conv1D(filters=32, kernel_size=15, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(x1)
    x1 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x1)
    x1 =Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(x1)
    x1 = SeqSelfAttention(
        kernel_regularizer=regularizers.l2(1e-3),
        bias_regularizer=regularizers.l2(1e-4),
        attention_regularizer_weight=1e-4,
        attention_activation='sigmoid',
        name='Attention')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1, activation='sigmoid')(x1)
    model=Model(input=inp1,outputs=x1)
    return model

def Model2():
    inp2 = Input(shape=(300,28))
    x2 =Conv1D(filters=128, kernel_size=5, padding='same', activation='relu', kernel_initiaizer='zero',kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(inp2)
    x2 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x2)
    x2 =Conv1D(filters=32, kernel_size=10, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(x2)
    x2 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x2)
    x2 =Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(x2)
    x2 = SeqSelfAttention(
        kernel_regularizer=regularizers.l2(1e-3),
        bias_regularizer=regularizers.l2(1e-4),
        attention_regularizer_weight=1e-4,
        attention_activation='sigmoid',
        name='Attention')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1, activation='sigmoid')(x2)
    model=Model(input=inp2,outputs=x2)
    return model

def Model3():
    inp3 = Input(shape=(300,28))
    x3 =Conv1D(filters=16, kernel_size=10, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(inp3)
    x3 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x3)
    x3 =Conv1D(filters=16, kernel_size=15, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(x3)
    x3 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x3)
    x3 =Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(x3)
    x3 = SeqSelfAttention(
        kernel_regularizer=regularizers.l2(1e-3),
        bias_regularizer=regularizers.l2(1e-4),
        attention_regularizer_weight=1e-4,
        attention_activation='sigmoid',
        name='Attention')(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(1, activation='sigmoid')(x3)
    model=Model(input=inp3,outputs=x3)
    return model

def Model4():
    inp4 = Input(shape=(300,16))
    x4 =Conv1D(filters=64, kernel_size=10, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(inp4)
    x4 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x4)
    x4 =Conv1D(filters=128, kernel_size=15, padding='same', activation='relu', kernel_initiaizer='zero', kernel_regularizer=regularizers.l2(1e-3),
               bias_regularizer=regularizers.l2(1e-4))(x4)
    x4 =MaxPooling1D(pool_size=2, strides=None, padding='valid')(x4)
    x4 =Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(x4)
    x4 = SeqSelfAttention(
        kernel_regularizer=regularizers.l2(1e-3),
        bias_regularizer=regularizers.l2(1e-4),
        attention_regularizer_weight=1e-4,
        attention_activation='sigmoid',
        name='Attention')(x4)
    x4 = Dropout(0.2)(x4)
    x4 = Flatten()(x4)
    x4 = Dense(1, activation='sigmoid')(x4)
    model=Model(input=inp4,outputs=x4)
    return model

def funciton(DATAPATH,OutputDir):

    maxlen = 300
    epochs = 100
    X_train,Y_train = createTrainData(DATAPATH)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('X_train shape:', X_train.shape)

    trainning_result = []
    testing_result = []
    k1=7
    k2=7

    X_train, X_test,Y_train, Y_test = train_test_split(X_train,Y_train, test_size=0.2, random_state=1337)
    X_train1, X_train2,X_train3, X_train4,Y_train = seq_matrix(X_train,Y_train,k1,k2)
    X_test1, X_test2,X_test3, X_test4 ,Y_test= seq_matrix(X_test,Y_test,k1,k2)
    X_train1 = sequence.pad_sequences(X_train1, maxlen=maxlen)
    X_train2 = sequence.pad_sequences(X_train2, maxlen=maxlen,dtype=float)
    X_train3 = sequence.pad_sequences(X_train3, maxlen=maxlen)
    X_train4 = sequence.pad_sequences(X_train4, maxlen=maxlen)
    X_test1 = sequence.pad_sequences(X_test1, maxlen=maxlen)
    X_test2 = sequence.pad_sequences(X_test2, maxlen=maxlen,dtype=float)
    X_test3 = sequence.pad_sequences(X_test3, maxlen=maxlen)
    X_test4 = sequence.pad_sequences(X_test4, maxlen=maxlen)


    print('X1 shape:', X_train1.shape)
    print('X2 shape:', X_train2.shape)
    print('X3 shape:', X_train3.shape)
    print('X4 shape:', X_train4.shape)

    model1 = Model1()
    model2 = Model2()
    model3 = Model3()
    model4 = Model4()

    model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
    model2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
    model3.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
    model4.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_check1 = ModelCheckpoint(filepath=OutputDir + "/model" + "1" + ".h5",
                                      monitor='val_loss', save_best_only=True)
    model_check2 = ModelCheckpoint(filepath=OutputDir + "/model" + "2" + ".h5",
                                      monitor='val_loss', save_best_only=True)
    model_check3 = ModelCheckpoint(filepath=OutputDir + "/model" + "3" + ".h5",
                                      monitor='val_loss', save_best_only=True)
    model_check4 = ModelCheckpoint(filepath=OutputDir + "/model" + "4" + ".h5",
                                      monitor='val_loss', save_best_only=True)
    reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8)
    hist1 = model1.fit(X_train1, Y_train,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 callbacks=[early_stopping,model_check1,reduct_L_rate],
                 validation_data=(X_test1, Y_test))
    hist2 = model2.fit(X_train2, Y_train,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 callbacks=[early_stopping,model_check2,reduct_L_rate],
                 validation_data=(X_test2, Y_test))
    hist3 = model3.fit(X_train3, Y_train,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 callbacks=[early_stopping,model_check3,reduct_L_rate],
                 validation_data=(X_test3, Y_test))

    hist4 = model4.fit(X_train4, Y_train,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 callbacks=[early_stopping,model_check4,reduct_L_rate],
                 validation_data=(X_test4, Y_test))

    prd_acc_1 = model1.predict(X_test1)
    prd_acc_2 = model2.predict(X_test2)
    prd_acc_3 = model3.predict(X_test3)
    prd_acc_4 = model4.predict(X_test4)
    final_preds = 0.4*prd_acc_1  +  0.3*prd_acc_2+ 0.2*prd_acc_3  +  0.1*prd_acc_4

    prd_lable = []
    for i in final_preds:
        if i > 0.5:
            prd_lable.append(1)
        else:
            prd_lable.append(0)
    prd_lable = np.array(prd_lable)
    obj = confusion_matrix(Y_test, prd_lable)
    tn,fp,fn,tp = obj.ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    Acc = (tp+tn)/(tp+tn+fp+fn)
    Mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    AUROC = metrics.roc_auc_score(Y_test, final_preds)


    print('***********************print final result*****************************')

    print('Sn : ', Sn)
    print('Sp : ', Sp)
    print('Acc : ',  Acc)
    print('Mcc : ',Mcc)
    print('AUROC: ',AUROC)
    FileNameR = '%s/Results.txt' % (args.output)
    Result_file = open(FileNameR, 'w')
    Result_file.write("Sn: %.2f%%\n" % (Sn * 100))
    Result_file.write("Sp: %.2f%%\n" % (Sp * 100))
    Result_file.write("Acc: %.2f%%\n" % (Acc * 100))
    Result_file.write("Mcc: %.2f%%\n" % (Mcc * 100))
    Result_file.write("AUROC: %.2f%%\n" % (AUROC * 100))
    Result_file.close()


if __name__ == "__main__":

    DATAPATH = './train.fa'
    OUTPATH = './output/'
    parser = argparse.ArgumentParser(description='Manual to the DHS')
    parser.add_argument('-i','--inputfile', type=str, help=' data', default=DATAPATH)
    parser.add_argument('-o','--output', type=str, help='output folder', default=OUTPATH)

    args = parser.parse_args()
    main()
