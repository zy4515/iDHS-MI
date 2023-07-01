#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function
import os
os.chdir('/data/home/zongyu/iDHS_MI')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import sys, argparse
from keras.models import load_model
import matplotlib as mpl
mpl.use('Agg')
import gc
from metrics_plot import *
import _pickle as cPickle
from keras.preprocessing import sequence
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, \
    roc_curve, roc_auc_score, auc, precision_recall_curve
from self_attention import SeqSelfAttention
import collections

parser = argparse.ArgumentParser(description='Manual to the DHS')
parser.add_argument('-p','--datapath', type=str, help=' data', required=True)
parser.add_argument('-a','--model1path', type=str, help='model folder', required=True)
parser.add_argument('-b','--model2path', type=str, help='model folder', required=True)
parser.add_argument('-c','--model3path', type=str, help='model folder', required=True)
parser.add_argument('-d','--model4path', type=str, help='model folder', required=True)
parser.add_argument('-o','--outpath', type=str, help='output file', required=True)
parser.add_argument('-t','--tissuse', type=str, help='tissuse', required=True)

args = parser.parse_args()


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

maxlen = 300
k1=7
k2=7
X_train, Y_train = createTrainData(args.datapath)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train1, X_train2,X_train3, X_train4,Y_train = seq_matrix(X_train,Y_train,k1,k2)
X_train1 = sequence.pad_sequences(X_train1, maxlen=maxlen)
X_train2 = sequence.pad_sequences(X_train2, maxlen=maxlen,dtype=float)
X_train3 = sequence.pad_sequences(X_train3, maxlen=maxlen)
X_train4 = sequence.pad_sequences(X_train4, maxlen=maxlen)


model1 = load_model(args.model1path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
model2 = load_model(args.model2path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
model3 = load_model(args.model3path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
model4 = load_model(args.model4path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
testing_result = []

prd_1 = model1.predict(X_train1)
prd_2 = model2.predict(X_train2)
prd_3 = model3.predict(X_train3)
prd_4 = model4.predict(X_train4)
final_preds = 0.4*prd_1  +  0.3*prd_2+ 0.2*prd_3  +  0.1*prd_4
prd_lable = []
for i in final_preds:
    if i > 0.5:
        prd_lable.append(1)
    else:
        prd_lable.append(0)
prd_lable = np.array(prd_lable)
obj = confusion_matrix(Y_train, prd_lable)
print(obj)
tn,fp,fn,tp = obj.ravel()
print(tp,fp,tn,fn)
Sn = tp / (tp + fn)
Sp = tn / (tn + fp)
Acc = (tp + tn) / (tp + tn + fp + fn)
Mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
AUROC = metrics.roc_auc_score(Y_train, final_preds)

print('***********************print final result*****************************')
print('Sn : ', Sn)
print('Sp : ', Sp)
print('Acc : ', Acc)
print('Mcc : ', Mcc)
print('AUROC: ' + str(AUROC))
FileNameR = '%s/%stestResults.txt' % (args.outpath,args.tissuse)
Result_file = open(FileNameR, 'w')
Result_file.write("Sn: %.2f%%\n" % (Sn * 100))
Result_file.write("Sp: %.2f%%\n" % (Sp * 100))
Result_file.write("Acc: %.2f%%\n" % (Acc * 100))
Result_file.write("Mcc: %.2f%%\n" % (Mcc * 100))
Result_file.write("AUROC: %.2f%%\n" % (AUROC * 100))
Result_file.close()