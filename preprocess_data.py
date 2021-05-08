#!/usr/bin/env python
# coding: utf-8

import pandas as pd
#import urllib.request
from tqdm import tqdm

#swag = urllib.request.urlopen('https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv')
swag = pd.read_csv('swag.csv')
cols = swag.columns.tolist()

label_map = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}

train = pd.read_json("train.json")

res = []
for idx, row in tqdm(train.iterrows()):
    content = row['Content']
    questions = row['Questions']
    for question in questions:
        q_id = question['Q_id']
        choices = question['Choices']
        answer = question['Answer']
        question = question['Question']
        modified_choices = ["", "", "", ""]
        for choice_idx, choice in enumerate(choices):
            modified_choices[choice_idx] = choice[2:]
        label = label_map[answer]
        ## Hard-code for swag format!
        res.append(("", 
                    "",
                    q_id, 
                    "", 
                    content, 
                    question, 
                    "", 
                    modified_choices[0], 
                    modified_choices[1], 
                    modified_choices[2], 
                    modified_choices[3], 
                    label))

test = pd.read_json('validation.json')

res = []
for idx, row in tqdm(test.iterrows()):
    content = row['Content']
    questions = row['Questions']
    for question in questions:
        q_id = question['Q_id']
        choices = question['Choices']
        question = question['Question']
        modified_choices = ["", "", "", ""]
        for choice_idx, choice in enumerate(choices):
            modified_choices[choice_idx] = choice[2:]
        ## Hard-code for swag format!
        res.append(("", 
                    "",
                    q_id, 
                    "", 
                    content, 
                    question, 
                    "", 
                    modified_choices[0], 
                    modified_choices[1], 
                    modified_choices[2], 
                    modified_choices[3],
                    0))
        
df = pd.DataFrame(res, columns=cols)        
df_test = pd.DataFrame(res, columns=cols)

df.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)
