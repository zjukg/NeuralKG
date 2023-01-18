import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict as ddict
from IPython import embed


class RuleDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.rule_p, self.rule_q, self.rule_r, self.confidences, self.tripleNum = [], [], [], [], []
        with open(os.path.join(args.data_path, 'groudings.txt')) as f:
            for line in f.readlines():
                token = line.strip().split('\t')
                for i in range(len(token)):
                    token[i] = token[i].strip('(').strip(')')

                iUnseenPos = int(token[0])
                self.tripleNum.append(iUnseenPos)

                iFstHead = int(token[1])
                iFstTail = int(token[3])
                iFstRelation = int(token[2])
                self.rule_p.append([iFstHead, iFstRelation, iFstTail])

                iSndHead = int(token[4])
                iSndTail = int(token[6])
                iSndRelation = int(token[5])
                self.rule_q.append([iSndHead, iSndRelation, iSndTail])

                if len(token) == 8:
                    confidence = float(token[7])
                    self.rule_r.append([0, 0, 0])

                else:
                    confidence = float(token[10])
                    iTrdHead = int(token[7])
                    iTrdTail = int(token[9])
                    iTrdRelation = int(token[8])
                    self.rule_r.append([iTrdHead, iTrdRelation, iTrdTail])
                self.confidences.append(confidence)
            self.len = len(self.confidences)
            self.rule_p = torch.tensor(self.rule_p).to(self.args.gpu)
            self.rule_q = torch.tensor(self.rule_q).to(self.args.gpu)
            self.rule_r = torch.tensor(self.rule_r).to(self.args.gpu)
            self.confidences = torch.tensor(self.confidences).to(self.args.gpu)
            self.tripleNum = torch.tensor(self.tripleNum).to(self.args.gpu)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.rule_p[idx], self.rule_q[idx], self.rule_r[idx]), self.confidences[idx], self.tripleNum[idx]


class RuleDataLoader(DataLoader):
    def __init__(self, args):
        dataset = RuleDataset(args)
        super(RuleDataLoader, self).__init__(
            dataset=dataset,
            batch_size=int(dataset.__len__()/args.num_batches),
            shuffle=args.shuffle)