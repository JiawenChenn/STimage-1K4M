---
layout: article
title: Gene expression prediction
sidebar:
  nav: document_nav
permalink: docs/11-gene-exp-prediction
---

In this document, we evaluated the performance of the zero-shot embedding on gene expression prediction task. We fit a panelized linear regression model to predict gene expression using image embeddings. This involved training a simple linear model on 80\% of the Human_Breast_Andersson_10142021_ST_H1 data, then the hyperparameter is selected based on the performance on the 10\% test data. Then we fit the linear regression using all data from Human_Breast_Andersson_10142021_ST_H1 and test the performance on Human_Breast_Andersson_10142021_ST_H2.

```python
#py3.10
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import glob
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import pandas as pd
import scanpy as sc


def eval_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'pearson_corr': pearson_corr}

def run_prediction(train_x, train_y, test_x, test_y, val_x, val_y, random_state=0, alpha=0.1):
    reg = Ridge(solver='lsqr',
            alpha=alpha, 
            random_state=random_state, 
            fit_intercept=False, 
            max_iter=1000)
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    val_y = val_y.to_numpy()
    reg.fit(train_x, train_y)
    test_pred = reg.predict(test_x)
    train_pred = reg.predict(train_x)
    val_pred = reg.predict(val_x)
    train_matrics = eval_metrics(train_y, train_pred)
    test_metrics = eval_metrics(test_y, test_pred)
    val_metrics = eval_metrics(val_y, val_pred)
    return {'train_mse': train_matrics['mse'], 'test_mse': test_metrics['mse'], 'val_mse': val_metrics['mse'],
            'train_pearson_corr': train_matrics['pearson_corr'], 'test_pearson_corr': test_metrics['pearson_corr'], 'val_pearson_corr': val_metrics['pearson_corr']}



meta = pd.read_csv('./meta/meta_all_gene.csv')


gene_exp = pd.read_csv('./ST/gene_exp/Human_Breast_Andersson_10142021_ST_H1_count.csv', index_col=0)
adata = sc.AnnData(gene_exp)
adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_genes(adata, min_cells=1)
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)
gene_exp = pd.DataFrame(adata.X,index=adata.obs.index,columns=adata.var.index)



model_record_best = []
for model_name in ['CLIP','PLIP','uni']:
    zero_shot = pd.read_csv(f'./zero_shot_embedding/{model_name}_human_image_feature.csv',index_col=0)
    overlap_index = gene_exp.index.intersection(zero_shot.index)
    gene_exp_subset = gene_exp.loc[overlap_index]
    embedding = zero_shot.loc[overlap_index]
    all_records_dataset = []
    for k in range(5):
        np.random.seed(k)
        train_index = np.random.choice(gene_exp_subset.index, int(0.8*gene_exp_subset.shape[0]), replace=False)
        test_index = np.random.choice(list(set(gene_exp_subset.index)-set(train_index)), int(0.1*gene_exp_subset.shape[0]), replace=False)
        val_index = list(set(gene_exp_subset.index)-set(train_index)-set(test_index))
        train_y = gene_exp_subset.loc[train_index].ERBB2
        train_x = embedding.loc[train_index].to_numpy()
        test_y = gene_exp_subset.loc[test_index].ERBB2
        test_x = embedding.loc[test_index].to_numpy()
        val_y = gene_exp_subset.loc[val_index].ERBB2
        val_x = embedding.loc[val_index].to_numpy()
        all_records = []
        for alpha in [1.0, 0.1, 0.01, 0.001,0.0001]:
            metrics = run_prediction(train_x, train_y, test_x, test_y, val_x, val_y, alpha = alpha)
            metrics["alpha"] = alpha
            metrics["test_on"] = 'split'+str(k)
            metrics["model_name"] = model_name
            all_records.append(metrics)
        all_records_dataset.extend(all_records)
        #
    all_records_dataset_df = pd.DataFrame(all_records_dataset)
    best_alpha = all_records_dataset_df.groupby('alpha')['val_mse'].mean().idxmin()
    mean_mse = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_mse'].mean()
    std_mse = all_records_dataset_df[all_records_dataset_df['alpha'] == best_alpha]['test_mse'].std()
    record_best = {'model_name':model_name,'best_alpha':best_alpha,'mean_mse':mean_mse,'std_mse':std_mse}
    model_record_best.append(record_best)
    print(model_name)

model_record_best_df = pd.DataFrame(model_record_best)
model_record_best_df.to_csv('./prediction/zero_shot_result.csv',index=False,sep='\t')


# train on all data and save prediction on H2

coord_H2 = pd.read_csv('./ST/coord/Human_Breast_Andersson_10142021_ST_H2_coord.csv', index_col=0)

model_record_best = []
for model_name in ['CLIP','PLIP','uni']:
    zero_shot = pd.read_csv(f'./zero_shot_embedding/{model_name}_human_image_feature.csv',index_col=0)
    overlap_index = gene_exp.index.intersection(zero_shot.index)
    gene_exp_subset = gene_exp.loc[overlap_index]
    embedding = zero_shot.loc[overlap_index]
    # train on H1
    reg = Ridge(solver='lsqr',
        alpha=model_record_best_df[model_record_best_df['model_name'] == model_name]['best_alpha'].values[0], 
        random_state=0, 
        fit_intercept=False, 
        max_iter=1000)
    train_y = gene_exp_subset.ERBB2
    train_x = embedding.to_numpy()
    reg.fit(train_x, train_y)
    # predict on H2
    test_x = zero_shot.loc[coord_H2.index].to_numpy()
    test_pred = reg.predict(test_x)
    test_pred = pd.DataFrame(test_pred,index=coord_H2.index,columns=['ERBB2'])
    test_pred.to_csv(f'./prediction/zero_shot_result_{model_name}.csv',sep='\t')

gene_H2 = pd.read_csv('./ST/gene_exp/Human_Breast_Andersson_10142021_ST_H2_count.csv', index_col=0)
adata = sc.AnnData(gene_H2)
adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_genes(adata, min_cells=1)
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)
gene_H2 = pd.DataFrame(adata.X,index=adata.obs.index,columns=adata.var.index)
gene_H2 = gene_H2.ERBB2
gene_H2.to_csv('./prediction/truth_H2.csv',sep='\t')
```