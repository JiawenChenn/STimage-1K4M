---
layout: article
title: Image representation learning
sidebar:
  nav: document_nav
permalink: docs/08-embedding_score
---
To evaluate the enhancement in image representations achieved by the fine-tuned models, we utilized pathologist-annotated brain layers as benchmarks to calculate several cluster quality metrics, including the Silhouette score, the Calinski-Harabasz index, and the Davies-Bouldin index.

```python
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score


model_path = glob.glob('./embedding/withanno/finetune*Human_Brain_Maynard_02082021_Visium*')
anno_temp = pd.read_csv('./anno_forlp/Human_Brain_Maynard_02082021_Visium_anno.csv',sep='\t',index_col=0)
anno_temp = anno_temp[anno_temp.V2 != '']
anno_temp = anno_temp[anno_temp.V2.notnull()]
anno_temp = anno_temp[anno_temp.V2 != 'undetermined']
anno_temp = anno_temp[anno_temp.V2 != 'Exclude']

sil_score = []
ch_score = []
db_score = []

model = []
slide_all = []
for j in range(len(model_path)):
    embedding_all = pd.read_csv(model_path[j],index_col=0)
    model_name = model_path[j].replace('./embedding/withanno/','')
    model_name = model_name.replace('_withanno.csv','')
    print(model_name)
    for slide in anno_temp.sample_name.unique():
        anno_temp_slide = anno_temp[anno_temp.sample_name == slide]
        index_keep = anno_temp_slide.index.intersection(embedding_all.index)
        embedding = embedding_all.loc[index_keep]
        anno = anno_temp_slide.loc[index_keep]
        sil_score.append(silhouette_score(embedding,anno.V2))
        ch_score.append(calinski_harabasz_score(embedding,anno.V2))
        db_score.append(davies_bouldin_score(embedding,anno.V2))
        model.append(model_name)
        slide_all.append(slide)

model_name_zero = ['CLIP','PLIP','uni']
for model_name in model_name_zero:
    embedding1 = pd.read_csv(f'./zero_shot_embedding/{model_name}_human_image_feature.csv',index_col=0)
    embedding2 = pd.read_csv(f'./zero_shot_embedding/{model_name}_mouse_image_feature.csv',index_col=0)
    embedding_all = pd.concat([embedding1,embedding2])
    print(model_name)
    for slide in anno_temp.sample_name.unique():
        anno_temp_slide = anno_temp[anno_temp.sample_name == slide]
        index_keep = anno_temp_slide.index.intersection(embedding_all.index)
        embedding = embedding_all.loc[index_keep]
        anno = anno_temp_slide.loc[index_keep]
        sil_score.append(silhouette_score(embedding,anno.V2))
        ch_score.append(calinski_harabasz_score(embedding,anno.V2))
        db_score.append(davies_bouldin_score(embedding,anno.V2))
        model.append(model_name)
        slide_all.append(slide)
        
result = pd.DataFrame({'model':model,'silhouette_score':sil_score,
                       'Calinski-Harabasz_score':ch_score,'Davies-Bouldin_score':db_score,'slide':slide_all})

result.to_csv('silhouette_score_32.csv',index=False,sep='\t')
```