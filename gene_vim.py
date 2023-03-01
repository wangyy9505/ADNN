
import pickle

import torch
from torch import nn
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients

from sklearn.model_selection import train_test_split


class ClinicalBenefitCLF(nn.Module):

    def __init__(
        self,
        gene_num,
        drop_p
    ):
        super(ClinicalBenefitCLF, self).__init__()
        # embedding
        self.embed = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(gene_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(p=drop_p),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        # classifier
        self.clf = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        embed_x = self.embed(x)
        return self.clf(embed_x)


# torch.dtype
torch.set_default_dtype(torch.float32)
device = torch.device('cuda:1')

# read data and pheno
with open('pre_data.pkl', 'rb') as f:
    data = pickle.load(f)

data1, y1 = data['data2'], data['y2']

# split data
trainx, testx, trainy, testy = train_test_split(
    data1, y1, test_size=0.2, random_state=1234, shuffle=True, stratify=y1
)

testx = torch.FloatTensor(testx.values)
testy = torch.LongTensor(testy)

model = ClinicalBenefitCLF(gene_num=trainx.shape[1], drop_p=0.05)

with open('adv_wt.pkl', 'rb') as f:
    wt = pickle.load(f)

model.load_state_dict(wt)
model.eval()

# IG计算变量重要性
ig = IntegratedGradients(model)
testx = testx.requires_grad_()
ig_gene = ig.attribute(testx, target=1, return_convergence_delta=False)
ig_gene = pd.DataFrame(ig_gene.detach().numpy())
genes = pd.read_csv('cogenes.csv')
ig_gene.columns = genes.gene.values
ig_gene_vim = pd.DataFrame({'vim': ig_gene.apply(np.sum, axis=0)})
ig_gene_vim.loc[:, 'vim_abs'] = ig_gene_vim.vim.abs().values
ig_gene_vim.loc[:, 'vim_std'] = ig_gene.apply(np.std, axis=0)
ig_gene_vim = ig_gene_vim.sort_values(by='vim_abs', ascending=False)
ig_gene_vim.to_csv('gene_vim/integradient.csv')
