
import os
import csv
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-input", type=str, required=True)
parser.add_argument("-output", type=str, required=True)
args = parser.parse_args()


swag = pd.read_csv("swag.csv")
cols = swag.columns.tolist()

label_map = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}

test = pd.read_json(args.input)

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
            
df_test = pd.DataFrame(res, columns=cols)

df_test.to_csv('test.csv', index=False)



os.system("python3 run.py --model_name_or_path './macbert-large' --do_predict --max_seq_length 512 --test_file test.csv --output_dir 'output' --gradient_accumulation_steps 8 --per_device_eval_batch_size 16 --overwrite_output")


test_results = pd.read_csv('output/test_results.txt')
labels=test_results['label']
label=[]
for item in labels:
    label.append(item)
ids=df_test['fold-ind']
id=[]
for item in ids:
    id.append(item)

sub=open(args.output,'w',encoding='utf-8')
writer=csv.writer(sub)
writer.writerow(['id','label'])
for i in range(len(label)):
    writer.writerow([id[i],label[i]])
sub.close()





