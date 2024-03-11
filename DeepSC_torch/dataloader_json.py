from config import Config
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import torch
import json
import re
import nltk

pd.set_option("max_colwidth", None)

class Dataset(Dataset):
    def __init__(self, config, train_or_test, transform=None, target_transform=None):
        if train_or_test == 'train':
            self.data = load_json2list(config, 'train')
        elif train_or_test == 'test':
            self.data = load_json2list(config, 'dev')
        else:
            ValueError("Invalid value. Train or Test should be given.")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]['context']
        return data

    def batch_sample(self, batch_size):
        idx = torch.randint(0, len(self.data), [batch_size,1])
        data = [self.data[x]['context'] for x in idx]
        return data



class IterableDataset(IterableDataset):

    def __init__(self, train_or_test, transform=None):
        if train_or_test == 'train':
            self.data = pd.read_csv(config.paths['data_path'] + "train_data.csv", sep='\t', header=None, iterator=True, chunksize=1)
        elif train_or_test == 'test':
            self.data = pd.read_csv(config.paths['data_path'] + "test_data.csv", sep='\t', header=None, iterator=True, chunksize=1)
        else:
            ValueError("Invalid value. Train or Test should be given.")
        self.transform = transform

    def __iter__(self):
        for line in self.data:
            line = line['text'].item()
            yield line

def load_json2list(config, train_or_test):
    with open(config.paths['data_path'] + train_or_test + '-v2.0.json') as f:
        json_object = json.load(f)

    title_idx = 0
    paragraph_idx = 0
    context_id = 0
    context_list = []

    while True:
        try:
            title_itr = json_object['data'][title_idx]['title']
        except IndexError:
            break
        while True:
            try:
                context_itr = json_object['data'][title_idx]['paragraphs'][paragraph_idx]['context']
                # if context_itr.find('Warsaw Uprising Hill') > 0:
                #     print('a')

                sentences = nltk.sent_tokenize(context_itr)

                for context_sen_itr in sentences:
                    if len(context_sen_itr) > 0:
                        context_list.append({'title': title_itr, 'context_id': context_id, 'context': context_sen_itr})
            except IndexError:
                break
            paragraph_idx += 1
            context_id += 1

        title_idx += 1
        paragraph_idx = 0

    return context_list


if __name__ == '__main__':
    config = Config()
    ds = Dataset(config, 'train')
    print(ds.__len__())
    print(ds.__getitem__(0))

    ds_it = IterableDataset(config, 'train')
    print(next(ds_it))