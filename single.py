import os
import random
import numpy as np
import pandas as pd
import torch
from bert_classifier import EL_CodeBert
from fig import draw_bar

seed = 2022
random.seed(seed)
os.environ['PYHTONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data = pd.read_csv('./data/SMELL_train.csv')
valid_data = pd.read_csv('./data/SMELL_test.csv')
test_data  = pd.read_csv('./data/SMELL_test.csv')

classifier = EL_CodeBert(
        model_path='./codebert',
        tokenizer_path='./codebert',
        pretrained_model='./save/smell_codebertrnnatten.pt',
        max_len=256,
        n_classes=2,
        model_save_path='./save/smell_codebertrnnatten.pt'
)

texts = list(test_data['code'])
labels = list(test_data['label'])

print(texts[184])
prediction, alpha = classifier.single_predict(texts[184])
print(alpha)
key_name = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9', 'Layer 10', 'Layer 11', 'Layer 12', ]
key_values = alpha
draw_bar(key_name, key_values)