---
layout: article
title: Crop spot-level image
sidebar:
  nav: document_nav
permalink: docs/03-crop-image
---

To perform spot-level analysis, you need to crop the slide images into spot-level. We used the following code to crop the spot-level images. We note here although all the images are named as png, they are actully a mixture of jpg, tiff and png. We suggest you using `PIL.Image.open` since it can automatically detect image type.

```python
import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000

def crop_image_all(path,slide):
  # We note here that although
  temp_image = Image.open(f'{path}/image/{slide}.png')
  temp_image = temp_image.convert('RGB')
  coord = pd.read_csv(f'{path}/coord/{slide}_coord.csv')
  r = coord.r[0]
  for i in range(len(coord.index)):
      yaxis = coord.yaxis[i]
      xaxis = coord.xaxis[i]
      spot_name = coord.iloc[i,0]
      temp_image_crop = temp_image.crop((xaxis-r, yaxis-r, xaxis+r, yaxis+r))
      temp_image_crop.save(f"./crop/{spot_name}.png")
        #plt.scatter(xaxis,yaxis,c='r')



index = int(sys.argv[1])
meta = pd.read_csv('meta_all_gene.csv',sep='\t')

slide = meta.slide[index]
tech = meta.tech[index]

path=f'./{tech}/'
crop_image_all(path,slide)
```
