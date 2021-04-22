import os
import argparse
import pandas as pd


from sklearn.metrics import accuracy_score, f1_score, classification_report
from prettyprinter import cpprint


answer_df = pd.read_csv('./data/test.csv')
submission_df = pd.read_csv('./prediction/submission.csv')

labels = os.listdir('./fruits-360/Test')
answer = answer_df.label.tolist()
submission = submission_df.label.tolist()

print(f'\n\t\t     Accuracy : {accuracy_score(answer, submission):.5f}\n')
print(f'\t\t     F1 Score : {f1_score(answer, submission, average="macro"):.5f}\n')

for i, (ans, subm) in enumerate(zip(answer, submission)):
    answer[i] = labels[ans]
    submission[i] = labels[subm]

cpprint('*==================Classificaion Report==================*')
print(classification_report(answer, submission))
