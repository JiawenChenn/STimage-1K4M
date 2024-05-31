import sys
import pandas as pd
import numpy as np
import scanpy as sc
import os

path='.'

def overlap_hvg(tissue_type):
    # gene exp of overlapping gene
    gene_exp_slide = pd.read_csv(f'{path}/{tissue_type}_count_overlap.csv',sep=',',index_col=0,header=None)
    gene_exp_slide.columns = ['gene'+str(i) for i in range(gene_exp_slide.shape[1])]
    adata = sc.AnnData(gene_exp_slide)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    # sort genes by highly_variable
    adata_hvg = adata[:, adata.var.highly_variable]
    hvg = adata_hvg.X
    hvg = pd.DataFrame(hvg)
    hvg.index = adata_hvg.obs.index
    hvg.to_csv(f'./all/{tissue_type}_count_overlap_hvg.csv',sep=',')

tissue_type = sys.argv[1]
overlap_hvg(tissue_type)

