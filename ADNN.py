
import copy
import pickle

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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


class ReverseGrad(Function):

    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x) * 1.

    @staticmethod
    def backward(ctx, grad_outputs):
        return (grad_outputs * -1. * ctx.weight), None


class AdvClf(nn.Module):

    def __init__(self, clfmodel, drop_p):
        super(AdvClf, self).__init__()
        self.embed = clfmodel.embed
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

    def forward(self, x, weight):
        embed_x = self.embed(x)
        rev_x = ReverseGrad.apply(embed_x, weight)
        return self.clf(rev_x)


class MlpDataSet(Dataset):

    def __init__(self, x, y1, y2=None):
        super(MlpDataSet, self).__init__()
        self.x, self.y1, self.y2 = x, y1, y2

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, index):
        if self.y2 is None:
            return self.x[index, :], self.y1[index]
        else:
            return self.x[index, :], self.y1[index], self.y2[index]


def train_model(
    model, adv_model, dataloader, epochs, early_stop, learn_rate, weight_decay,
    momentum, T_max, device, grad_weight
):
    model_wt = copy.deepcopy(model.state_dict())
    best_auc = .0
    best_loss = np.inf
    no_improve = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay,
        momentum=momentum
    )
    optimizer.add_param_group({
        'params': adv_model.clf.parameters(),
        'names': 'adv_clf'
    })
    optimizer_step = CosineAnnealingLR(
        optimizer=optimizer, T_max=T_max, eta_min=0.0001
    )
    # optimizer_step = StepLR(
    #     optimizer=optimizer, step_size=lr_step_size, gamma=lr_gamma
    # )

    for _ in tqdm(range(epochs), leave=False):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                adv_model.train()
            else:
                model.eval()

            epoch_loss = .0
            epoch_loss_adv = .0
            y_true, y_pred = [], []
            adv_true, adv_pred = [], []
            with torch.set_grad_enabled(phase == 'train'):
                for x, y, adv_y in dataloader[phase]:
                    x, y, adv_y = x.to(device), y.to(device), adv_y.to(device)
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == 'train':
                        adv_output = adv_model(x, grad_weight)
                        adv_loss = criterion(adv_output, adv_y)
                        optimizer.zero_grad()
                        (loss + adv_loss).backward()
                        optimizer.step()
                        epoch_loss_adv += adv_loss.item() * len(adv_y)
                        adv_true.append(adv_y)
                        adv_pred.append(adv_output)
                    epoch_loss += loss.item() * len(y)
                    y_true.append(y)
                    y_pred.append(output)

            with torch.no_grad():
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                auc = roc_auc_score(y_true, y_pred[:, 1])
                epoch_loss /= len(y_true)
                if phase == 'train':
                    adv_true = torch.cat(adv_true, dim=0).cpu().numpy()
                    adv_pred = torch.cat(adv_pred, dim=0).cpu().numpy()
                    adv_auc = roc_auc_score(adv_true, adv_pred[:, 1])

                if phase == 'valid':
                    if any((auc > best_auc, epoch_loss < best_loss)):
                        best_auc = np.max((auc, best_auc))
                        best_loss = np.min((epoch_loss, best_loss))
                        model_wt = copy.deepcopy(model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
        optimizer_step.step()
        if no_improve == early_stop:
            break
    with torch.no_grad():
        model.load_state_dict(model_wt)
        model.eval()
        # 测试集
        test_true, test_pred = [], []
        for x, y in dataloader['test']:
            x = x.to(device)
            output = model(x)
            test_true.append(y)
            test_pred.append(output)
        test_true = torch.cat(test_true, dim=0).numpy()
        test_pred = torch.cat(test_pred, dim=0).cpu().numpy()
        test_auc = roc_auc_score(test_true, test_pred[:, 1])
        test_result = pd.DataFrame(
            {'ytrue': test_true, 'ypred': test_pred[:, 1]}
        )
        # 超参数选择集
        hyper_true, hyper_pred = [], []
        for x, y in dataloader['hyper']:
            x = x.to(device)
            output = model(x)
            hyper_true.append(y)
            hyper_pred.append(output)
        hyper_true = torch.cat(hyper_true, dim=0).numpy()
        hyper_pred = torch.cat(hyper_pred, dim=0).cpu().numpy()
        hyper_auc = roc_auc_score(hyper_true, hyper_pred[:, 1])
        return test_result, test_auc, hyper_auc, model_wt


def main():
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_p', default=[0.05])
    parser.add_argument('--learn_rate', type=list, default=[0.01])
    parser.add_argument('--epochs', type=list, default=[300])
    parser.add_argument('--early_stop', type=list, default=[50])
    parser.add_argument('--weight_decay', type=list, default=[1e-4, 1e-3])
    parser.add_argument('--momentum', type=list, default=[0.9])
    parser.add_argument('--T_max', type=list, default=[25, 50, 75])
    parser.add_argument('--grad_weight', default=[0.5, 0.7, 0.9])
    parser.add_argument('--bs', type=list, default=[8])
    args = parser.parse_args()

    # Grid research
    param_grid = list(ParameterGrid({
        'drop_p': args.drop_p,
        'learn_rate': args.learn_rate,
        'epochs': args.epochs,
        'early_stop': args.early_stop,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'T_max': args.T_max,
        'grad_weight': args.grad_weight,
        'bs': args.bs
    }))
    # pd.DataFrame(param_grid).to_csv('param_grid.csv', index=None)

    # torch.dtype
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:1')

    # read data and pheno
    with open('pre_data.pkl', 'rb') as f:
        data = pickle.load(f)

    data1, y1 = data['data2'], data['y2']
    data2, y2 = data['data1'], data['y1']

    # split data
    trainx, testx, trainy, testy = train_test_split(
        data1, y1, test_size=0.2, random_state=1234, shuffle=True, stratify=y1
    )
    trainx, validx, trainy, validy = train_test_split(
        trainx, trainy, test_size=len(testy),
        random_state=1234, shuffle=True, stratify=trainy
    )
    trainx, hyperx, trainy, hypery = train_test_split(
        trainx, trainy, test_size=len(testy),
        random_state=1234, shuffle=True, stratify=trainy
    )

    train_advy = torch.LongTensor([0] * trainx.shape[0] + [1] * data2.shape[0])
    trainx = torch.FloatTensor(pd.concat([trainx, data2], axis=0).values)
    trainy = torch.LongTensor(np.concatenate([trainy, y2], axis=0))

    validx = torch.FloatTensor(validx.values)
    validy = torch.LongTensor(validy)
    valid_advy = torch.LongTensor([0] * len(validy))

    hyperx = torch.FloatTensor(hyperx.values)
    hypery = torch.LongTensor(hypery)

    testx = torch.FloatTensor(testx.values)
    testy = torch.LongTensor(testy)

    test_aucs, hyper_aucs = [], []
    for epo, params in tqdm(
        enumerate(param_grid), total=len(param_grid), leave=False
    ):
        drop_p = params['drop_p']
        learn_rate = params['learn_rate']
        epochs = params['epochs']
        early_stop = params['early_stop']
        weight_decay = params['weight_decay']
        momentum = params['momentum']
        T_max = params['T_max']
        grad_weight = params['grad_weight']
        bs = params['bs']

        torch.manual_seed(1234)
        # dataloader
        dataloader = {
            'valid': DataLoader(
                MlpDataSet(validx, validy, valid_advy), batch_size=bs
            ),
            'test': DataLoader(
                MlpDataSet(testx, testy), batch_size=bs
            ),
            'hyper': DataLoader(
                MlpDataSet(hyperx, hypery), batch_size=bs
            )
        }
        if trainx.size(0) % bs == 1:
            dataloader['train'] = DataLoader(
                MlpDataSet(trainx, trainy, train_advy), batch_size=bs,
                shuffle=True, drop_last=True
            )
        else:
            dataloader['train'] = DataLoader(
                MlpDataSet(trainx, trainy, train_advy), batch_size=bs,
                shuffle=True
            )

        # model
        model = ClinicalBenefitCLF(data1.shape[1], drop_p=drop_p).to(device)
        adv_model = AdvClf(clfmodel=model, drop_p=drop_p).to(device)

        # trainmodel and return result
        test_result, test_auc, hyper_auc, model_wt = train_model(
            model=model, adv_model=adv_model, dataloader=dataloader,
            epochs=epochs, early_stop=early_stop, learn_rate=learn_rate,
            weight_decay=weight_decay, momentum=momentum, T_max=T_max,
            device=device, grad_weight=grad_weight
        )
        test_aucs.append(test_auc)
        hyper_aucs.append(hyper_auc)

        if epo == 0:
            test_result.to_csv('control/adv_test_pred.csv', index=None)
            with open('adv_wt.pkl', 'wb') as f:
                pickle.dump(model_wt, f)
        elif np.argmax(hyper_aucs) == len(hyper_aucs) - 1:
            test_result.to_csv('control/adv_test_pred.csv', index=None)
            with open('adv_wt.pkl', 'wb') as f:
                pickle.dump(model_wt, f)
        # with open('gridsearch/parma' + str(epo) + '.pkl', 'wb') as f:
        #     pickle.dump(total_result, f)

        result = pd.DataFrame({
            'param': range(len(test_aucs)), 'test': test_aucs,
            'hyper': hyper_aucs
        })
        if epo == len(param_grid) - 1:
            params = pd.DataFrame(param_grid)
            result.loc[:, 'param_idx'] = range(len(param_grid))
            result = pd.concat([params, result], axis=1)
        result.to_csv('control/Adv_AUC.csv',index=None)


if __name__ == '__main__':
    main()
