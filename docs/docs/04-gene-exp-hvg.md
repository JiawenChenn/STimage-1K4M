---
layout: article
title: Gene expression processing
sidebar:
  nav: document_nav
permalink: docs/04-gene-exp-hvg
---

For integration with gene expression, we reduced the dimension of gene expression by selecting highly variable genes (HVGs). We explored two strategies: highly variable genes (HVG) selected separately from each slide, and HVGs selected from overlapping genes across slides (overlap-HVG)

#### 1. HVG

##### (1) Find HVG of each slide
```python
#python
import pandas as pd
import scanpy as sc
import sys


meta = pd.read_csv('meta_all_gene.csv',sep='\t')
path = '.'

# load data
i = int(sys.argv[1])
slide = meta['slide'][i]
gene_exp_slide = pd.read_csv(f'{path}/{meta.tech[i]}/gene_exp/{meta.slide[i]}_count.csv',sep=',',index_col=0)
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
```

##### (2) Combined HVGs

Suppose we want to combine all HVGs for human brain slides, we first extract the slide names for all human brain slides.

```python
#python
data = meta.loc[(meta['species'] == 'human') & (meta['tissue'] == 'brain'),:]
data.slide.to_csv(f'human_brain_slide.csv',index=False,header=False)
```

Then we combine them together.

```bash
#bash
type='human_brain'
path='./'
rm ${type}_count_hvg.csv
while read slide;do
    if [ -f ${path}/HVG/${slide}_count_hvg.csv ]; then
        cat ${path}/HVG/${slide}_count_hvg.csv | tail -n +2 >> ./${type}_count_hvg.csv
    fi
    echo $slide
done < ${type}_slide.csv
```

#### 2. Overlap-HVG

Here we showcase how to find Overlap-HVG in human brain.

##### (1) Find overlap gene

```python
# python
# here we find the overlap genes in human brain
data = meta.loc[(meta['species'] == 'human') & (meta['tissue'] == 'brain'),:]
data.index = range(len(data.index))

for i in range(len(data.index)):
    gene_exp_slide = pd.read_csv(f'{path}/{data.tech[i]}/gene_exp/{data.slide[i]}_count.csv',sep=',',nrows=1,index_col=0)
    if i == 0:
        gene_name_overlap = gene_exp_slide.columns
    else:
        gene_name_overlap = gene_name_overlap.intersection(gene_exp_slide.columns)
    print(i)

pd.DataFrame(gene_name_overlap).to_csv('human_brain_gene.csv',index=False,header=False)
```

##### (2) Extract gene expression for each slide
Here we extract human brain overlap genes for all human slides.

```python
meta = pd.read_csv('meta_all.csv',sep='\t')
path = '.'

species = 'human'
tissue = 'all'
gene_list = 'human_brain_gene'

if tissue == 'all':
    data = meta.loc[(meta['species'] == species),:]
    data.slide.to_csv(f'{species}_slide.csv',index=False,header=False)
else:
    data = meta.loc[(meta['species'] == species) & (meta['tissue'] == tissue),:]
    data.slide.to_csv(f'{species}_{tissue}_slide.csv',index=False,header=False)


data.index = range(len(data.index))

for index in range(data.index):
    slide = data['slide'][index]
    gene_exp_slide = pd.read_csv(f'{path}/{data.tech[index]}/gene_exp/{data.slide[index]}_count.csv',sep=',',index_col=0)
    overlap_gene = pd.read_csv(f'{gene_list}.csv',header=None)
    gene_exp_slide = gene_exp_slide.loc[:,overlap_gene[0]]
    gene_exp_slide.to_csv(f'./overlap-hvg/{data.slide[index]}_{gene_list}.csv',sep=',')
```

##### (3) Combine all gene expression

```bash
#bash
type='human'
path='./overlap-hvg/'
rm /proj/yunligrp/users/jiawen/spatial_omics_data/organized_data_gene_exp/all/${type}_count_overlap.csv
while read slide;do
    if [ -f ${path}/${slide}_${type}_gene.csv ]; then
        cat ${path}/${slide}_${type}_gene.csv | tail -n +2 >> ${type}_count_overlap.csv
    else
        echo $slide
    fi
done < ${type}_slide.csv

```

##### (4) Extract HVG from combined gene expression

```python
# python
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import os

path='.'

def overlap_hvg(tissue_type):
    # Here we take input of combined gene expression of one tissue type
    # see below for how to generate this file
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
```