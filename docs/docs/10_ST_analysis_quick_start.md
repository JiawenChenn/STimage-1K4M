---
layout: article
title: ST analysis quick start
sidebar:
  nav: document_nav
permalink: docs/10_ST_analysis_quick_start
---

For a quick and easy start of ST data analysis, here we recommend the use of three popular analysis packages. It is also very useful to check the tutorials of these packages for a better understanding of the common tasks and data structure.
- [Seruat](https://satijalab.org/seurat/): R toolkit for single cell genomics.
- [Scanpy](https://scanpy.readthedocs.io/en/stable/index.html): Single-Cell analysis in Python.
- [Squidpy](https://squidpy.readthedocs.io/en/stable/index.html): Spatial Single Cell Analysis in Python.

Here we include code for how to load STimage-1K4M data as annData for easy use of Scanpy and Squidpy.

```python
#python
import pandas as pd
import scanpy as sc
import sys


meta = pd.read_csv('meta_all_gene.csv',sep='\t')
path = '.'

# load data
i = 1
slide = meta['slide'][i]
gene_exp_slide = pd.read_csv(f'{path}/{meta.tech[i]}/gene_exp/{meta.slide[i]}_count.csv',sep=',',index_col=0)
adata = sc.AnnData(gene_exp_slide)
adata.var_names_make_unique()
```

Then, with adata, we can perform standard data preprocessing like normalization and log1p transformation.

```python
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
```

Then you can follow the pipeline in Squidy for spatial data analysis. For example, you can analyze spatial variable genes follow this [pipeline](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_hne.html#image-features).