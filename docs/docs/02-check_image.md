---
layout: article
title: Check spot location
sidebar:
  nav: document_nav
permalink: docs/02-check_image
---

We manually check the spot location on each slide. The processed coodinate file include yaxis, xaxis and the radius for each spot.

```python
import sys
from typing import List,Dict
import pandas as pd
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000


# check the coordinate of the spot on the slide

def check_image_all(path,slide):
    temp_image = Image.open(f'{path}/image/{slide}.png')
    temp_image = temp_image.convert('RGB')
    coord = pd.read_csv(f'{path}/coord/{slide}_coord.csv')
    r = coord.r[0] # radius
    plt.clf()
    plt.imshow(temp_image)
    for i in range(len(coord.index)):
        yaxis = coord.yaxis[i]
        xaxis = coord.xaxis[i]
        plt.plot([xaxis-r,xaxis+r],[yaxis-r,yaxis-r],c='lime',alpha=0.5)
        plt.plot([xaxis-r,xaxis+r],[yaxis+r,yaxis+r],c='lime',alpha=0.5)
        plt.plot([xaxis-r,xaxis-r],[yaxis-r,yaxis+r],c='lime',alpha=0.5)
        plt.plot([xaxis+r,xaxis+r],[yaxis-r,yaxis+r],c='lime',alpha=0.5)
    # crop plt
    plt.axis('off')
    plt.axis([coord.xaxis.min()-r*2,coord.xaxis.max()+r*2,coord.yaxis.max()+r*2,coord.yaxis.min()-r*2])
    plt.savefig(f'./check_image_all/{slide}.png',dpi=100,bbox_inches='tight',pad_inches = 0)


meta = pd.read_csv('meta_all_gene.csv',sep='\t')

for index in range(len(meta)):
    slide = meta.slide[index]
    tech = meta.tech[index]
    # set your directory to downloaded STimage-1K4M
    path=f'./{tech}/'
    check_image_all(path,slide)
```