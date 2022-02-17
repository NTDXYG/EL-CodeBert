import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby


df = pd.read_csv("COMMENT_train.csv")
code_list = df['text'].tolist()
label_list = df['label'].tolist()

code_len_list = [len(str(code).split()) for code in code_list]
print(len(list(set(label_list))))
#
# count = 0
# for i in range(len(label_list)):
#     if(label_list[i] == 0):
#         count+=1
# print(count, len(label_list)-count)

def getBili(num, demo_list):
    s = 0
    for i in range(len(demo_list)):
        if(demo_list[i] < num):
            s += 1
    print('<'+str(num)+'比例为'+str(s/len(demo_list)))

from numpy import *

b = mean(code_len_list)
c = median(code_len_list)
counts = np.bincount(code_len_list)
d = np.argmax(counts)
print('平均值'+str(b))
print('众数'+str(d))
print('中位数'+str(c))

getBili(32,code_len_list)
getBili(64,code_len_list)
getBili(128,code_len_list)
getBili(256,code_len_list)
# getBili(512,code_len_list)