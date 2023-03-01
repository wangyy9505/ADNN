
import pickle

# import torch
import numpy as np
import pandas as pd

# from torch_geometric.utils import to_dense_adj

# from reactome_net import generate_maps


def data_process():
    data1 = pd.read_csv('rawdata/GSE163211/GSE163211_Exp.csv', index_col=0)
    pheno1 = pd.read_csv('rawdata/GSE163211/GSE163211_Pheno.csv', index_col=0)
    all(data1.index.values == pheno1.index.values)

    data2 = pd.read_csv('rawdata/GSE48452/GSE48452_Exp.csv', index_col=0)
    pheno2 = pd.read_csv('rawdata/GSE48452/GSE48452_Pheno.csv', index_col=0)
    all(data2.index.values == pheno2.samp.values)

    data3 = pd.read_csv('rawdata/GSE89632/GSE89632_Exp.csv', index_col=0)
    pheno3 = pd.read_csv('rawdata/GSE89632/GSE89632_Pheno.csv', index_col=0)
    all(data3.index.values == pheno3.samp.values)

    cogenes = np.intersect1d(data1.columns.values, data2.columns.values)
    cogenes = np.intersect1d(data3.columns.values, cogenes)

    data1 = data1.loc[:, cogenes]
    data2 = data2.loc[:, cogenes]
    data3 = data3.loc[:, cogenes]

    pd.DataFrame({'gene': cogenes}).to_csv('cogenes.csv', index=None)

    y1 = np.zeros(pheno1.shape[0])
    y1[pheno1.loc[:, 'nafld.stage'].values != 'Normal'] = 1

    y2 = np.zeros(pheno2.shape[0])
    y2[pheno2.group.values == 'Steatosis'] = 1
    y2[pheno2.group.values == 'Nash'] = 1

    y3 = np.ones(pheno3.shape[0])
    y3[pheno3.group.values == 'HC'] = 0

    # string = pd.read_csv('rawdata/string.csv')
    # string = string.loc[:, ['gene_name.x', 'gene_name.y']]
    # string.columns = ['x', 'y']
    # string_genes = np.unique(
    #     np.concatenate([string.x.unique(), string.y.unique()])
    # )
    # gene_id = pd.DataFrame({
    #     'gene': string_genes, 'id': range(len(string_genes))
    # })
    # string = pd.merge(string, gene_id, left_on='x', right_on='gene')
    # string = pd.merge(string, gene_id, left_on='y', right_on='gene')
    # string = torch.LongTensor(string[['id_x', 'id_y']].T.values)
    # dense_adj = to_dense_adj(string).squeeze().numpy()
    # dense_adj = pd.DataFrame(
    #     dense_adj, index=gene_id.gene.values, columns=gene_id.gene.values
    # )
    # cogenes = np.intersect1d(cogenes, dense_adj.index.values)
    # data1, data2 = data1.loc[:, cogenes], data2.loc[:, cogenes]
    # data3 = data3.loc[:, cogenes]
    # dense_adj = dense_adj.loc[cogenes, :]
    # dense_adj = dense_adj.loc[:, dense_adj.apply(np.sum, axis=0) != 0.]

    # maps = generate_maps(genes=dense_adj.columns.values, n_levels=n_levels)
    # maps = [dense_adj] + maps

    pre_data = {
        'data1': data1, 'y1': y1, 'data2': data2, 'y2': y2, 'data3': data3,
        'y3': y3
    }
    with open('pre_data.pkl', 'wb') as f:
        pickle.dump(pre_data, f)


if __name__ == '__main__':
    data_process()
