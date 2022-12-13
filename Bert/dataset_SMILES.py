import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

device = 'cuda'


class SmilesDataset(Dataset):
    def __init__(
            # self, opt, dataset, split, tokenizer, debug=False,
            # max_length=None, entity_max_length=None,
            # prompt_tokenizer=None, prompt_max_length=None
            self, opt, src_path
    ):
        super(SmilesDataset, self).__init__()
        self.data = []
        self.opt = opt
        # self.src_path = opt.data
        self.src_path = src_path

        self.vocab = self.load('data/USPTO-50k_no_rxn/USPTO-50k_no_rxn.vocab.txt')
        self.vocab['<mask>'] = len(self.vocab)
        self.vocab['<global>'] = len(self.vocab)
        self.get_data()

    def get_smiles(self):
        with open(self.src_path, 'r') as f:
            lines = f.readlines()
        return lines

    def get_tokens(self, smile):
        tokens = smile.strip().split(' ')
        return tokens

    def load(self, path):
        vocab_dict = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                vocab = line.strip().split('\t')
                vocab_dict[vocab[0]] = int(vocab[1])
            return vocab_dict

    def get_data(self):
        smiles = self.get_smiles()
        for smile in smiles:
            data = {}
            tokens = self.get_tokens(smile)
            tokens.insert(0, '<s>')
            tokens.insert(0, '<global>')
            tokens.append('</s>')
            nums_list = [self.vocab.get(i, self.vocab['<unk>']) for i in tokens]
            choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15), 1)] + 1
            y = nums_list.copy()
            weight = np.zeros(len(nums_list))
            for i in choices:
                rand = np.random.rand()
                weight[i] = 1
                if rand < 0.8:
                    nums_list[i] = self.vocab['<mask>']
                elif rand < 0.9:
                    r_num = int(np.random.rand() * (len(self.vocab)-4)) + 3
                    while nums_list[i] == r_num:
                            r_num = int(np.random.rand() * (len(self.vocab)-4)) + 3
                    nums_list[i] = r_num
            weight = weight.tolist()
            x = nums_list
            data['x'] = x
            data['weight'] = weight
            data['y'] = y
            self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class SmilesCollator(object):
    def __init__(
            # self, tokenizer, device, pad_entity_id, debug=False,
            # max_length=None, entity_max_length=None,
            # prompt_tokenizer=None, prompt_max_length=None,
            # use_amp=False
            self, max_length=None
    ):
        super(SmilesCollator, self).__init__()
        self.max_length = max_length

    def add_pad(self, data):
        data_len = len(data['x'])
        data_copy = dict()
        data_copy['x'] = data['x'].copy()
        data_copy['weight'] = data['weight'].copy()
        data_copy['y'] = data['y'].copy()

        for i in range(self.max_length - data_len):
            data_copy['x'].append(1)
            data_copy['weight'].append(0)
            data_copy['y'].append(1)
        return data_copy

    def __call__(self, data_batch):
        x = []
        weight = []
        y = []
        valid_lens = []
        for i, data in enumerate(data_batch):
            if len(data['x']) < self.max_length:
                data = self.add_pad(data)
                x.append(data['x'])
                weight.append(data['weight'])
                y.append(data['y'])
                valid_lens.append(len(data['x']))
            # elif len(data['x']) == self.max_length:
            #     x.append(data['x'])
            #     weight.append(data['weight'])
            #     y.append(data['y'])
                # data_batch[i] = data
        x = torch.tensor(x, dtype=torch.int64).to(device)
        weight = torch.tensor(weight, dtype=torch.int32).to(device)
        y = torch.tensor(y, dtype=torch.int64).to(device)
        valid_lens = torch.tensor(valid_lens, dtype=torch.int32, device=device)

        input_batch = {}

        input_batch['x'] = x
        input_batch['weight'] = weight
        input_batch['y'] = y
        input_batch['valid_lens'] = valid_lens
        return input_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get saved data/model path")
    parser.add_argument('--src_path', '-src_path', type=str, default='data/USPTO-50k_no_rxn/src-train.txt')
    # parser.add_argument('--model_path', '-model_path', type=str, default='experiments/USPTO-50k_no_rxn_Best_model')
    # parser.add_argument('--model_path', '-model_path', type=str, default='experiments/USPTO-50k_no_rxn_pretrain_train_full_word2vec_100000')
    args = parser.parse_args()
    train_data = SmilesDataset(args, args.src_path)

    data_collator = SmilesCollator(
        max_length=256
    )
    dataloader = DataLoader(
        train_data,
        batch_size=64,
        collate_fn=data_collator,
    )

    input_max_len = 0
    entity_max_len = 0
    i = 0
    for epoch in range(3):
        i = 0
        for batch in tqdm(dataloader):
            x = batch['x']
            seq_len = x.shape[1]
            if i == 2500:
                print(batch)
            i += 1
            # print(i)
            # x = batch['y']
            # w = torch.nonzero(batch['weight'])
            # x = x.reshape(-1)
            # masked_X = [x[idx[0]*256+idx[1]] for idx in w]
            # masked_X = torch.tensor(masked_X).unsqueeze(0)
            # print(masked_X)
    print(123)


