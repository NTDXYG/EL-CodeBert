import os
import random
import pandas as pd
from bert_classifier import *

seed = 2022
random.seed(seed)
os.environ['PYHTONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_df = pd.read_csv('./data/CODE_train.csv')
eval_df = pd.read_csv('./data/CODE_test.csv')
test_df = pd.read_csv('./data/CODE_test.csv')

train_df.columns = ["input_text", "target_text"]
eval_df.columns = ["input_text", "target_text"]
test_df.columns = ["input_text", "target_text"]

# first fine-tune CodeBert
classifier = CodeBertClassifier(
        model_path='./codebert',
        tokenizer_path='./codebert',
        max_len=256,
        n_classes=19,
        epochs=5,
        model_save_path='./save/code_codebert.pt',
        batch_size=64,
        learning_rate=2e-5
)

# then fine-tune the others
# classifier = EL_CodeBert(
#         model_path='./codebert',
#         tokenizer_path='./codebert',
#         pretrained_model='./save/code_codebert.pt',
#         max_len=256,
#         n_classes=19,
#         epochs=15,
#         model_save_path='./save/code_codebertrnnatten.pt',
#         batch_size=64,
#         learning_rate=5e-4
# )

# classifier = CodeBertClassifierRNN(
#         model_path='./codebert',
#         tokenizer_path='./codebert',
#         pretrained_model='./save/smell_codebert.pt',
#         max_len=256,
#         n_classes=2,
#         epochs=15,
#         model_save_path='./save/smell_codebertrnn.pt',
#         batch_size=64,
#         learning_rate=5e-4
# )

# classifier = CodeBertClassifierAtten(
#         model_path='./codebert',
#         tokenizer_path='./codebert',
#         pretrained_model='./save/smell_codebert.pt',
#         n_classes=2,
#         max_len=256,
#         epochs=15,
#         model_save_path='./save/smell_codebertatten.pt',
#         batch_size=64,
#         learning_rate=5e-4
# )

classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text'])
    )

classifier.train()

texts = list(test_df['input_text'])
labels = list(test_df['target_text'])

predictions = []
for i in tqdm(range(len(texts))):
        predictions.append(classifier.predict(texts[i]))

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
acc = accuracy_score(labels, predictions)
print(f'acc: {acc}, precision: {precision}, recall: {recall}, f1score: {f1score}')