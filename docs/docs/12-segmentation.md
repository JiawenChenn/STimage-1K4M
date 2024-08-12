---
layout: article
title: Evaluate nuclei segmentation
sidebar:
  nav: document_nav
permalink: docs/12-segmentation
---

In this document, we showcase how to use STimage-1K4M to evaluate nuclei segmentation and cell type assignment. We note here that we don't have pixel level cell type information. With the help of gene expression data, we can use cell type deconvolution to infer the cell type proportion (see a summary for deconvolution methods [here](https://pubmed.ncbi.nlm.nih.gov/35753702/)). Here we used RCTD (PMID 33603203) to infer the cell type proportion of Human_Breast_Andersson_10142021_ST_H1 with the help of an external scRNA-seq reference (PMID 34493872). 

Here, the cell segmentation method we employed is [CellViT](https://github.com/TIO-IKIM/CellViT?tab=readme-ov-file). We compare the performance of pretrained CellViT-256 and CellViT-SAM to the RCTD inferred cell type proportion.

First, use this code to convert STimage-1K4M image to OpenSlide image.

convert_tiff.py

```python
from PIL import Image
import sys
import pyvips
import numpy as np
from openslide import OpenSlide
import tifffile

Image.MAX_IMAGE_PIXELS = None
img_path = sys.argv[1]
img = np.array(Image.open(img_path))

if img.shape[0] == 3:
    img = np.transpose(img, axes=(1, 2, 0))
elif img.shape[2] == 4: # RGBA to RGB
    img = img[:,:,:3]

pyvips_img = pyvips.Image.new_from_array(img)
save_path=sys.argv[2]

pixel_size = float(sys.argv[3]) # 1 pixel = ? mu m

pyvips_img.tiffsave(
    save_path, 
    bigtiff=False, 
    pyramid=True, 
    tile=True, 
    tile_width=256, 
    tile_height=256, 
    compression='jpeg', 
    resunit=pyvips.enums.ForeignTiffResunit.CM,
    xres=1. / (pixel_size * 1e-4),
    yres=1. / (pixel_size * 1e-4))
```

```bash
python3 convert_tiff.py \
./ST/image/Human_Breast_Andersson_10142021_ST_H1.png \
Human_Breast_Andersson_10142021_ST_H1.tif \
0.688

tifftools set -y -s ImageDescription  "Aperio Fake |AppMag = 20|MPP = 0.68" Human_Breast_Andersson_10142021_ST_H1.tif
```

Then we can use CellViT to do cell segmentation.

```bash
cd CellViT
python3 ./preprocessing/patch_extraction/main_extraction.py --config preprocessing_example.yaml


python3 ./cell_segmentation/inference/cell_detection.py \
  --model ./models/pretrained/CellViT-SAM-H-x20.pth\
  --gpu 0 \
  --geojson \
  --magnification 20. \
  process_wsi \
  --wsi_path Human_Breast_Andersson_10142021_ST_H1.tif \
  --patched_slide_path ./preprocessing/Human_Breast_Andersson_10142021_ST_H1

cp ./preprocessing/Human_Breast_Andersson_10142021_ST_H1 ./preprocessing0/Human_Breast_Andersson_10142021_ST_H1

python3 ./cell_segmentation/inference/cell_detection.py \
  --model ./models/pretrained/CellViT-256-x20.pth\
  --gpu 0 \
  --geojson \
  --magnification 20. \
  process_wsi \
  --wsi_path Human_Breast_Andersson_10142021_ST_H1.tif \
  --patched_slide_path ./preprocessing0/Human_Breast_Andersson_10142021_ST_H1
```

preprocessing_example.yaml
```
# dataset paths
wsi_paths: Human_Breast_Andersson_10142021_ST_H1.tif
output_path: ./preprocessing
wsi_extension: tif
# basic setups
patch_size: 1024
patch_overlap: 6.25
target_mag: 20
processes: 8
overwrite: True

# macenko stain normalization
normalize_stains: True

# finding patches
min_intersection_ratio: 0.05
```

After cell segmentation, we compare the spot-level cell type proportion. In order to do that, we need to aggregate the cells segmented by CellViT in each spot.

```python

import json
from pathlib import Path
from typing import List, Tuple, Union

import argparse
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

import pandas as pd
import json


with open('./preprocessing/Human_Breast_Andersson_10142021_ST_H1/cell_detection/cell_detection.json') as json_data:
    data = json.load(json_data)

# load image
image_full = Image.open('./Human_Breast_Andersson_10142021_ST_H1.tif')
image_full = image_full.convert('RGB')

# get the cell types of the cells and plot the contour of the cells
with open('./preprocessing0/Human_Breast_Andersson_10142021_ST_H1/cell_detection/cells.geojson', encoding='utf-8') as f:
    gj = json.load(f)

gj = pd.DataFrame(gj) 

# plot the cells with cell type
fig, ax = plt.subplots(1, 1, figsize=(100, 100))
ax.imshow(image_full)

for i in range(5):
    contour = gj.iloc[i,]['geometry']['coordinates']
    color = gj.iloc[i,]['properties']['classification']['color']
    color = (color[0]/255, color[1]/255, color[2]/255)
    cell_type = gj.iloc[i,]['properties']['classification']['name']
    for j in range(len(contour)):
        contour_cell = np.array(contour[j])[0]
        ax.plot(contour_cell[:,0], contour_cell[:,1], color=color,label = cell_type)

fig.savefig('.segmentation/cell.png')


# get the cell types of the cells
with open('./preprocessing0/Human_Breast_Andersson_10142021_ST_H1/cell_detection/cell_detection.geojson', encoding='utf-8') as f:
    gj = json.load(f)

gj = pd.DataFrame(gj)
data_organized = pd.DataFrame()

for i in range(5):
    temp = pd.DataFrame(gj.iloc[i,]['geometry']['coordinates'])
    temp['cell_type'] = gj.iloc[i,]['properties']['classification']['name']
    data_organized = pd.concat([data_organized,temp])

# organize data
data_organized.columns = ['x','y','cell_type']

coord = pd.read_csv('./ST/coord/Human_Breast_Andersson_10142021_ST_H1_coord.csv', index_col=0)

# only keep the spot within the Cell-ViT inferred region
coord_keep = coord[coord['xaxis']-coord['r'] >= data_organized.x.min()]
coord_keep = coord_keep[coord_keep['xaxis']+coord_keep['r'] <= data_organized.x.max()]
coord_keep = coord_keep[coord_keep['yaxis']-coord_keep['r'] >= data_organized.y.min()]
coord_keep = coord_keep[coord_keep['yaxis']+coord_keep['r'] <= data_organized.y.max()]

index_celltype = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
inferred = np.zeros((coord_keep.shape[0],5))

# aggregate cell number in each spot
for i in range(coord_keep.shape[0]):
    xaxis = coord_keep.iloc[i,]['xaxis']
    yaxis = coord_keep.iloc[i,]['yaxis']
    radius = coord_keep.iloc[i,]['r']
    cell_in = data_organized[(data_organized['x']-xaxis)**2 + (data_organized['y']-yaxis)**2 <= radius**2]
    if cell_in.shape[0] == 0:
        continue
    else:
        for j in range(5):
            inferred[i,j] = cell_in[cell_in['cell_type'] == index_celltype[j]].shape[0]
    print(i)


coord_keep['Neoplastic'] = inferred[:,0]
coord_keep['Inflammatory'] = inferred[:,1]
coord_keep['Connective'] = inferred[:,2]
coord_keep['Dead'] = inferred[:,3]
coord_keep['Epithelial'] = inferred[:,4]

coord_keep.to_csv('.segmentation/cell_type_256.csv')

# Then repeat the same code for SAM based model.
```