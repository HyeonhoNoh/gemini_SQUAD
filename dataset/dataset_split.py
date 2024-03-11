import pandas as pd
from sklearn.model_selection import train_test_split
import csv

pd.set_option("max_colwidth", None)
df = pd.read_json("train-v2.0.json")


import json
with open('train-v2.0.json') as f:
    json_object = json.load(f)

title_idx = 0
paragraph_idx = 0
context_list = []

while True:
    try:
        title_itr = json_object['data'][title_idx]['title']
    except IndexError:
        break

    while True:
        try:
            context_itr = json_object['data'][title_idx]['paragraphs'][paragraph_idx]['context']
            context_list.append({'title': title_itr, 'context': context_itr})
        except IndexError:
            break

        paragraph_idx += 1

    title_idx += 1
    paragraph_idx = 0

    print([title_idx, paragraph_idx])


# Save the training and test datasets to separate CSV files
train_data = "train_data.csv"

train_df.to_csv(train_csv, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\", sep="\t", header=False, index=False)
test_df.to_csv(test_csv,  quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\", sep="\t", header=False, index=False)

print("Data successfully split and saved to training_data.csv and test_data.csv.")