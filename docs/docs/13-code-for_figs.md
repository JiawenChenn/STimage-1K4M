---
layout: article
title: Code for reproducing figures
sidebar:
  nav: document_nav
permalink: docs/13-code-for_figs
---

In this document, we include code for reproducing figures in the paper.


### Figure 1c

```r
library(data.table)
library(ggplot2)    
library(egg)
library(ggsci)
library(ggrepel)
library(tidyverse)
library(cowplot)
library(ggthemes)
library(dplyr)
library(forcats)


blank_theme <- theme_minimal()+
  theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.border = element_blank(),
  panel.grid=element_blank(),
  axis.ticks = element_blank(),
  plot.title=element_text(size=14, face="bold")
)



meta = fread('./meta/meta_all_gene.csv')
df = table(meta$tech)
df = data.table(tech = names(df), value = as.numeric(df))
df2 <- df %>% 
mutate(csum = rev(cumsum(rev(value))), 
        pos = value/2 + lead(csum, 1),
        pos = if_else(is.na(pos), value/2, pos))

k1 = ggplot(df, aes(x = "" , y = value, fill = fct_inorder(tech))) +
geom_col(width = 1, color = 1) +
coord_polar(theta = "y") +
scale_fill_brewer(palette = "Pastel1") +
geom_label_repel(data = df2,
                aes(y = pos, label = value),
                size = 10, nudge_x = 1, show.legend = FALSE) +
guides(fill = guide_legend(title = "Technology")) +
theme_void()+theme(legend.position = "right",legend.title = element_text(size=25),legend.text = element_text(size=25))

df = data.table(group = c('ST','Visium','VisiumHD'), value = c(60145,2336306,1896744))
df2 <- df %>% 
mutate(csum = rev(cumsum(rev(value))), 
        pos = value/2 + lead(csum, 1),
        pos = if_else(is.na(pos), value/2, pos))

k2 = ggplot(df, aes(x = "" , y = value, fill = fct_inorder(group))) +
geom_col(width = 1, color = 1) +
coord_polar(theta = "y") +
scale_fill_brewer(palette = "Pastel1") +
geom_label_repel(data = df2,
                aes(y = pos, label = value),
                size = 10, nudge_x = 1, show.legend = FALSE) +
guides(fill = guide_legend(title = "Tech (spot)")) +
theme_void()+theme(legend.position = "right",legend.title = element_text(size=25),legend.text = element_text(size=25))


species_temp = meta$species
species_temp[! species_temp %in% c('human','mouse','human & mouse')]='other'
df = table(species_temp)
df = data.table(group = names(df), value = as.numeric(df))
df2 <- df %>% 
mutate(csum = rev(cumsum(rev(value))), 
        pos = value/2 + lead(csum, 1),
        pos = if_else(is.na(pos), value/2, pos))

k3 = ggplot(df, aes(x = "" , y = value, fill = fct_inorder(group))) +
geom_col(width = 1, color = 1) +
coord_polar(theta = "y") +
scale_fill_brewer(palette = "Pastel1") +
geom_label_repel(data = df2,
                aes(y = pos, label = value),
                size = 10, nudge_x = 1, show.legend = FALSE) +
guides(fill = guide_legend(title = "Species")) +
theme_void()+theme(legend.position = "right",legend.title = element_text(size=25),legend.text = element_text(size=25))


tissue = meta$tissue
tissue[! tissue %in% names(sort(-table(meta$tissue)))[1:10]]='other'
df = table(tissue)
df = data.table(group = names(df), value = as.numeric(df))
df2 <- df %>% 
mutate(csum = rev(cumsum(rev(value))), 
        pos = value/2 + lead(csum, 1),
        pos = if_else(is.na(pos), value/2, pos))

k4 = ggplot(df, aes(x = "" , y = value, fill = fct_inorder(group))) +
geom_col(width = 1, color = 1) +
coord_polar(theta = "y") +
scale_fill_tableau(palette = "Tableau 20") +
geom_label_repel(data = df2,
                aes(y = pos, label = value),
                size = 10, nudge_x = 1, show.legend = FALSE) +
guides(fill = guide_legend(title = "Tissue")) +
theme_void()+theme(legend.position = "right",legend.title = element_text(size=25),legend.text = element_text(size=25))

k = plot_grid(k1,k2,k3,k4,nrow=1)
```


### Figure 3

Code to generate tSNE embedding:
```python
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
import glob
now = datetime.now()


anno_path_list = glob.glob(f'./annotation/Human_Brain_Maynard*')
#anno_path_list = glob.glob('/proj/yunligrp/users/jiawen/st_data_paper/figure/table1/anno_forlp/*')


index = int(sys.argv[1])
model_list = glob.glob(f'./embedding/withanno/finetune_*_Human_Brain_Maynard_02082021*')

anno_name = anno_path_list[index]
anno_name_save = anno_name.replace('./annotation/','').replace('_anno.csv','')


for model_path in model_list:
    model_name = model_path.replace('./embedding/withanno/','').replace('_withanno.csv','')
    data = pd.read_csv(model_path,index_col=0,header=0)
    anno = pd.read_csv(anno_name,header=0,index_col=0)
    overlap_index = list(set(data.index.tolist()).intersection(set(anno.index.tolist())))
    data_anno = data.loc[overlap_index,:]
    anno = anno.loc[overlap_index,:]
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data_anno.to_numpy())
    data_tsne_pd = pd.DataFrame(data_tsne,index=data_anno.index)
    data_tsne_pd.to_csv(f'./figure3/tsne_byanno/{model_name}_{anno_name_save}_tsne.csv')
```

Code for plotting:

```R
# r/4.3.1

library(spatialLIBD)
library(jpeg)
library(data.table)
library(ggplot2)    
library(egg)
library(ggsci)
library(ggrepel)
library(tidyverse)
library(cowplot)
library(ggthemes)
library(tidyr)



lp_result = fread('linear_probing_result32.csv')
lp_result$model_name = gsub('finetune_','finetune ',lp_result$model_name,fixed=TRUE)
lp_result$model_name = gsub('Human_Brain_Maynard_02082021_Visium','spatialLIBD',lp_result$model_name,fixed=TRUE)
lp_result$model_name = gsub('human_brain','human-brain',lp_result$model_name,fixed=TRUE)
lp_result$model_name = gsub('human_breast','human-breast',lp_result$model_name,fixed=TRUE)
lp_result$model_name = gsub('mouse_brain','mouse-brain',lp_result$model_name,fixed=TRUE)
lp_result$model_name = gsub('overlap_hvg','overlap-hvg',lp_result$model_name,fixed=TRUE)
lp_result = lp_result %>% mutate(model_name_raw=model_name) %>% separate(model_name, into = c("model", "training_data",'geneset'), sep = "_")

lp_result_zero = fread('linear_probing_zero_result.csv') %>% rename(model_name_raw=model_name) 
lp_result_zero$training_data='zero-shot'
lp_result_zero$model=lp_result_zero$model_name_raw
lp_result_zero$geneset='zero-shot'
lp_result_all = rbind(lp_result,lp_result_zero,fill=TRUE)
lp_result_all$data_name_first20 = substr(lp_result_all$data_name,1,20)


lp_result_all$model_name_raw = factor(lp_result_all$model_name_raw,levels=c(sort(unique(lp_result$model_name_raw)),sort(unique(lp_result_zero$model_name_raw))))

lp_result_all = lp_result_all %>% filter(data_name=='Human_Brain_Maynard_02082021_Visium',training_data %in% c('spatialLIBD','zero-shot'))
lp_result_all$model_name_new = gsub('_spatialLIBD_',' ',lp_result_all$model_name_raw,fixed=TRUE)
lp_result_all$model_name_new = gsub('uni','UNI',lp_result_all$model_name_new,fixed=TRUE)
lp_result_all$model_name_new = factor(lp_result_all$model_name_new,levels=c('CLIP','finetune CLIP hvg','finetune CLIP overlap-hvg','finetune CLIP pca','PLIP','finetune PLIP hvg','finetune PLIP overlap-hvg','UNI'))

lp_result_all = lp_result_all %>% filter(geneset!='pca')

color_palette = c("#1f77b4","#729ECE","#aec7e8","#F28E2B","#FFBE7D","#FFDD71",'#9467BD')

p1=ggplot(lp_result_all) +
  geom_bar( aes(x=model_name_new, y=mean_wf1,fill=model_name_new), stat="identity",  alpha=1) +
  geom_errorbar( aes(x=model_name_new, ymin=mean_wf1-std_wf1, ymax=mean_wf1+std_wf1), colour="orange", alpha=0.9, size=0.4,width = 0.5)+
  #facet_grid(~data_name_first20)+
  theme_article()+
  scale_fill_manual(values=color_palette,name='Model name')+
  ylab('Mean F1')+
  xlab(NULL)+
  theme(axis.text.x = element_blank(),axis.ticks.x = element_blank())#,strip.text.x = element_text(size = 5))



# embedding
dlpfc_model_tsne = function(model_name,model_type='',gene_type='',show_label=FALSE){
    # generated tsne embedding using image embedding
    path = './figure3/tsne_byanno/'
    slides = c('151675')
    anno_all = c()
    tsne_all = c()
    for(slide in slides){
        tsne = fread(paste0(path,'/',model_name,'_','Human_Brain_Maynard_02082021_Visium_',slide,'_tsne.csv'),header=TRUE)
        colnames(tsne) = c('V1','tSNE1','tSNE2')
        anno = fread(paste0('./annotation/Human_Brain_Maynard_02082021_Visium_',slide,'_anno.csv'),header=TRUE)
        anno$slide = slide
        anno_all = rbind(anno_all,anno)
        tsne_all = rbind(tsne_all,tsne)
    }

    anno_all = anno_all %>% left_join(tsne_all)
    if(show_label){
        label_text_color='black'
        label_alpha=1
        legend_position='right'
    }else{
        label_text_color='white'
        label_alpha=0
        legend_position='none'
    }
    model_name = gsub('_',' ',model_name)
    k = ggplot(anno_all) + 
        geom_point(aes(x=tSNE1,y=tSNE2,color=V2),size=0.5,alpha=0.6) + 
        scale_color_tableau('Tableau 20',direction=-1,name='Brain\nlayer')+
        ylab(NULL)+
        xlab(paste0(model_type,' ',gene_type))+
        theme_article()+
        guides(colour = guide_legend(override.aes = list(size=5,alpha=label_alpha)))+
        theme(plot.title = element_text(hjust = 0.5),axis.title.x = element_text(size=10),
                legend.position=legend_position,
                axis.text.x = element_blank(),axis.text.y = element_blank(),axis.ticks.x = element_blank(),axis.ticks.y = element_blank())
    return(k)
}

k1=dlpfc_model_tsne('CLIP',show_label=TRUE)
legend = cowplot::get_legend(dlpfc_model_tsne('CLIP',show_label=TRUE))
k1=dlpfc_model_tsne('CLIP','CLIP')
k2=dlpfc_model_tsne('finetune_CLIP_Human_Brain_Maynard_02082021_Visium_hvg','finetune CLIP','hvg')
k3=dlpfc_model_tsne('finetune_CLIP_Human_Brain_Maynard_02082021_Visium_overlap_hvg','finetune CLIP','overlap-hvg')
k5=dlpfc_model_tsne('PLIP','PLIP')
k6=dlpfc_model_tsne('finetune_PLIP_Human_Brain_Maynard_02082021_Visium_hvg','finetune PLIP','hvg')
k7=dlpfc_model_tsne('finetune_PLIP_Human_Brain_Maynard_02082021_Visium_overlap_hvg','finetune PLIP','overlap-hvg')
k8=dlpfc_model_tsne('uni','UNI')
p2 = plot_grid(k1,k2,k3,k5,k6,k7,k8,legend,nrow=1,rel_widths=c(rep(1,7),0.4))


color_palette = c("#1f77b4","#729ECE","#aec7e8","#F28E2B","#FFBE7D","#FFDD71",'#9467BD')

sil_score = fread('silhouette_score_32.csv') %>% 
    rename(model_name=model) %>%
    mutate(model_name = gsub('finetune_','finetune ',model_name,fixed=TRUE)) %>%
    mutate(model_name = gsub('_Human_Brain_Maynard_02082021_Visium_',' ',model_name,fixed=TRUE)) %>%
    mutate(model_name = gsub('overlap_hvg','overlap-hvg',model_name,fixed=TRUE)) %>%
    mutate(model_name = gsub('uni','UNI',model_name,fixed=TRUE)) %>%
    filter(model_name != 'finetune CLIP pca') %>%
    mutate(model_name_new = factor(model_name,levels=c('CLIP','finetune CLIP hvg','finetune CLIP overlap-hvg','PLIP','finetune PLIP hvg','finetune PLIP overlap-hvg','UNI'))) %>%
    mutate(slide = gsub('Human_Brain_Maynard_02082021_Visium_','',slide,fixed=TRUE))%>%
    rename('Calinski-Harabasz'=`Calinski-Harabasz_score`,'Davies-Bouldin'=`Davies-Bouldin_score`,'Silhouette'=silhouette_score)

sil_score = melt(sil_score,id.vars=c('model_name','slide','model_name_new'))

p3=ggplot(sil_score) +
geom_boxplot( aes(x=model_name_new, y=value,color=model_name_new),  alpha=1,lwd=1) +
geom_point( aes(x=model_name_new, y=value,fill=slide),shape=21,size=1.5,alpha=0.9)+
facet_wrap(~variable,scales='free_y')+
theme_article()+
scale_color_manual(values=color_palette,name='Model name')+
scale_fill_igv(name='Dataset')+
ylab(NULL)+
xlab(NULL)+
guides(fill=guide_legend(ncol=2))+
theme(axis.text.x = element_blank(),axis.ticks.x = element_blank())

library(tiff)

img = readTIFF("./Visium/image/Human_Brain_Maynard_02082021_Visium_151675.png")

grob <- grid::rasterGrob(img, width = grid::unit(1, "npc"), height = grid::unit(1, "npc"))

anno1 = fread('./annotation/Human_Brain_Maynard_02082021_Visium_151675_anno.csv')
coord1 = fread('./Visium/coord/Human_Brain_Maynard_02082021_Visium_151675_coord.csv')
coord1 = coord1 %>% left_join(anno1)



anno_brain = ggplot(coord1) + 
    geom_spatial(
        data = tibble::tibble(grob = list(grob)),
        aes(grob = grob),
        x = 0.5,
        y = 0.5
    )+
    geom_point(aes(x=xaxis,y=yaxis,color=V2),shape=21,size=0.15,alpha=0.7)+
    scale_color_tableau('Tableau 20',direction=-1)+
    xlim(0, ncol(img)) +
    ylim(nrow(img), 0) +
    theme_article()+
    coord_fixed(expand = FALSE)+
    theme(axis.text = element_blank(),axis.title = element_blank(),axis.ticks = element_blank(),legend.position = "none")


legend2 = cowplot::get_legend(p3)
f_top = plot_grid(p1+facet_grid(~'Mean F1')+theme(legend.position='none',strip.text = element_text(size=10))+ylab(NULL),
               p3+theme(legend.position='none',strip.text = element_text(size=10)),
               plot_grid(cowplot::get_legend(p1),legend2[3],ncol=1),
               anno_brain,
               labels=c('a','b','','c'),nrow=1,rel_widths=c(0.2,0.8,0.3,0.5))
blank = ggplot()+theme_void()
k = plot_grid(f_top,
              plot_grid(blank,p2+theme(axis.title.x = element_text(size=10)),
              nrow=1,rel_widths=c(0.015,1)),
              nrow=2,labels=c('','d'),rel_heights=c(1,1))
ggsave('figure3.png',k,bg='white',width=13,height=5,dpi=300)

```


### Figure 4
```python
import re
import sys
from typing import List,Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageEnhance
from skimage.segmentation import clear_border
from skimage import measure, color, io
from torchvision import transforms
import pandas as pd
import torch as t
from torch.utils.data import Dataset
from scipy.stats import rankdata


import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2


import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

import glob
import requests

import scanpy as sc

Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000




def plot_image(dataset_name,slide,tech,data_size,ax,point_size=10):
    path = './'
    temp_image = Image.open(f'{path}/{tech}/image/{dataset_name}{slide}.png')
    temp_image = temp_image.convert('RGB')
    coord = pd.read_csv(f'{path}/{tech}/coord/{dataset_name}{slide}_coord.csv',index_col=0,header=0)
    anno = pd.read_csv(f'./annotation/{dataset_name}{slide}_anno.csv',index_col=0,header=0)
    anno = anno.loc[~anno.index.duplicated(keep='first'),:]
    coord = pd.concat([coord,anno],axis=1)
    coord.index = range(len(coord.index))
    ax.imshow(temp_image)
    sns.scatterplot(x=coord.xaxis,y=coord.yaxis,hue=coord.iloc[:,3].tolist(),ax=ax,s=point_size)
    #sns.move_legend(ax, "center right",bbox_to_anchor=(1.2, 0.5),frameon=False)
    ax.axis('off')
    ax.set_title(f'{dataset_name} (n={data_size})')
    ax.legend(loc="lower right",markerscale =20/point_size)


plt.clf()
fig,axs = plt.subplots(3,3,figsize=(25,25))
plot_image('Human_Brain_Maynard_02082021_Visium','_151676','Visium',12,axs[0][0],5)
plot_image('Human_Breast_Andersson_10142021_ST','_G2','ST',8,axs[0][1],50)
plot_image('Human_Breast_10X_06232020_Visium_Block_A_Section_1','','Visium',1,axs[0][2],10)
plot_image('Human_Breast_10X_06092021_Visium','','Visium',1,axs[1][0],10)
plot_image('GSE193460','_GSM5808054','Visium',4,axs[1][1],10)
plot_image('Human_Prostate_Erickson_08102022_Visium','_Patient_1_H1_2','Visium',7,axs[1][2],10)
plot_image('GSE175540','_GSM5924030','Visium',23,axs[2][0],10)
plot_image('GSE213688','_GSM6592049','Visium',14,axs[2][1],10)
plot_image('Mouse_Brain_10X_06232020_Visium_Sagittal_Anterior_Section_1','','Visium',1,axs[2][2],10)
axs[2][2].legend().set_visible(False)

plt.tight_layout()

```