import pandas as pd
import scanpy as sc

import sys


meta = pd.read_csv('./meta/meta_all_gene.csv',sep='\t')
path = '.'

i = int(sys.argv[1])

slide = meta['slide'][i]
gene_exp_slide = pd.read_csv(f'{path}/{meta.tech[i]}/gene_exp/{meta.slide[i]}_count.csv',sep=',',index_col=0)

# hvg
adata = sc.AnnData(gene_exp_slide)
adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=1)
sc.experimental.pp.highly_variable_genes(adata, n_top_genes=128)

# sort genes by highly_variable
adata.var_names_make_unique()
hvg_list = adata.var['highly_variable_rank']
hvg_list = hvg_list.sort_values()
hvg_list = hvg_list.dropna()

adata_hvg = adata[:, hvg_list.index]
sc.pp.normalize_total(adata_hvg)
sc.pp.log1p(adata_hvg)
hvg = adata_hvg.X
hvg = pd.DataFrame(hvg)
hvg.index = adata_hvg.obs.index
hvg.to_csv(f'./HVG/{meta.slide[i]}_count_hvg.csv',sep=',')