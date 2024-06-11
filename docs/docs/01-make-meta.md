---
layout: article
title: Meta data cleaning
sidebar:
  nav: document_nav
permalink: docs/01-make-meta
---

This document holds code used for meta data cleaning, paticular for tissue and species. The cleaned meta data is avaliable in meta_all_gene.csv under meta folder on [GitHub](https://github.com/JiawenChenn/STimage-1K4M/blob/main/meta/meta_all_gene.csv) and in STimage-1K4M.

```python
import re
import pandas as pd
import numpy as np

# Manual map the tissue type
GEO_visium = pd.read_csv('./organized_data/code/meta/GEO_visium_meta.csv',sep='\t')

GEO_visium['species'] = GEO_visium['organism']
GEO_visium['species'] = GEO_visium['species'].replace({'Mus musculus':'mouse','Homo sapiens':'human',
    'Medicago truncatula':'plant',
    'Takifugu alboplumbeus':'fish','Sus scrofa':'pig','Xenopus laevis':'frog',
    'Mus':'mouse','Electrophorus voltai':'fish','Homo sapiens; Mus musculus':'human & mouse'})

GEO_visium['species'] = GEO_visium['species'].str.lower()

GEO_visium['tissue'] = GEO_visium['chara'].str.lower().str.extract(r'tissue: ([^,]+),')

GEO_visium.loc[GEO_visium.tissue.isna(),:]
GEO_visium['chara'][GEO_visium.tissue.isna()].tolist()
GEO_visium['GSE'][GEO_visium.tissue.isna()].unique()

GEO_visium['tissue'][GEO_visium.GSE=='GSE179572'] = 'brain'
GEO_visium['tissue'][GEO_visium.GSE=='GSE183456'] = 'kidney'
GEO_visium['tissue'][GEO_visium.GSE=='GSE188257'] = 'ovary'
GEO_visium['tissue'][GEO_visium.GSE=='GSE194329'] = 'malignant gliomas'
GEO_visium['tissue'][GEO_visium.GSE=='GSE192742'] = 'liver'
GEO_visium['tissue'][GEO_visium.GSE=='GSE153424'] = 'brain'
GEO_visium['tissue'][GEO_visium.GSE=='GSE188888'] = 'heart'
GEO_visium['tissue'][GEO_visium.GSE=='GSE203612'] = GEO_visium.loc[GEO_visium.GSE=='GSE203612','chara'].str.extract(r'organ: ([^,]+),')[0].tolist()
GEO_visium['tissue'][GEO_visium.GSE=='GSE206306'] = 'kidney'
GEO_visium['tissue'][(GEO_visium.GSE=='GSE180682') & (GEO_visium.tissue.isna())] = 'digit'

GEO_visium['tissue'][GEO_visium.GSE=='GSE217843'] = 'pancreas'
GEO_visium['tissue'][GEO_visium.GSE=='GSE248356'] = 'melanoma'

GEO_visium['tissue_raw'] = GEO_visium['tissue']

GEO_visium['tissue'] = GEO_visium.tissue.replace({'ccrcc tumor': 'kidney',
                                                'distal colon': 'colon',
                                                'brain (e13.5)': 'brain',
                                                'cardiac': 'heart',
                                                'gs3 breast cancer pdx': 'breast',
                                                'gastric cancer metastasis': 'stomach',
                                                'tnbc tumor tissue': 'breast',
                                                'human glioblastoma': 'glioblastoma',
                                                'tumor section': 'lacrimal gland',
                                                'healthy liver explant': 'liver',
                                                'atopic dermatitis lesional skin': 'skin',
                                                'coronal brain section': 'brain',
                                                'pdac tumor': 'pancreas',
                                                'liver biopsy': 'liver',
                                                'prostate cancer': 'prostate',
                                                'normal skin': 'skin',
                                                'subcutaneous tumor': 'skin',
                                                'idh mutant glioma': 'glioma',
                                                'malignant gliomas': 'glioma',
                                                'sagittal suture': 'brain',
                                                'oral mucosa': 'mouth',
                                                'human fetal pancreas': 'pancreas',
                                                'cscc': 'skin',
                                                'panin': 'pancreas',
                                                'ovarian cancer': 'ovary',
                                                'skin punch biopsy': 'skin',
                                                'skin wounds': 'skin',
                                                'atopic dermatitis non-lesional skin': 'skin',
                                                'pdac': 'pancreas',
                                                'adult liver': 'liver',
                                                'right coronary artery': 'heart',
                                                'human colorectal cancer': 'colon',
                                                'hepatocellular carcinoma': 'liver',
                                                'low grade ipmn': 'pancreas',
                                                'synovial joint': 'joint',
                                                'mc38 syngeneic tumor': 'colon',
                                                'liver containing metastases of crc origin': 'liver',
                                                'gastric cancer': 'stomach',
                                                'nonae liver explant': 'liver',
                                                'bone/cartilage': 'bone',
                                                'popliteal lymph nodes': 'lymph node',
                                                'lung tissue': 'lung',
                                                'tendon + injured area': 'tendon',
                                                'patellar tendon': 'tendon',
                                                'high grade ipmn': 'pancreas',
                                                'gan-kp tumor': 'stomach',
                                                'apap liver explant': 'liver',
                                                'oral squamous cell carcinoma (oscc)': 'mouth',
                                                'liver metastasis': 'liver',
                                                'trunk of pax2gfp mouse embryo': 'embryo',
                                                'coronary sample with plaque erosion': 'heart',
                                                'pancreatic cancer': 'pancreas',
                                                'mediastinal lymph node': 'lymph node',
                                                'heart ventricles collected 14 days after mi': 'heart',
                                                'skeletal muscle (quadriceps)': 'muscle',
                                                'melanoma tissue section': 'melanoma',
                                                '5 dpa regenerating lower arm': 'arm',
                                                'high grade ipmn with co-ocurring pdac': 'pancreas',
                                                'hcc': 'liver',
                                                'hippocampus': 'brain',
                                                'heart ventricles collected 3 days after mi': 'heart',
                                                'hs lesional skin': 'skin',
                                                'osf-associated oscc': 'mouth',
                                                'clm': 'liver',
                                                'crlm': 'liver',
                                                'normal control skin': 'skin',
                                                'heart ventricles collected 7 days after mi': 'heart',
                                                'sc31 breast cancer pdx': 'breast',
                                                'unwounded skin tissue': 'skin',
                                                'occc tumor': 'ovary',
                                                'heart ventricles collected at steady state': 'heart',
                                                'breast cancer': 'breast'})

GEO_visium['tissue'] = GEO_visium['tissue'].str.lower()
```

Here we also detect whether the study is related to cancer study by searning cancer, tumor, metastasis, and tme in study title, abstract and keywords.

```python
meta = pd.read_csv('./meta/meta_all_gene.csv',sep=',')
combine_text = meta['title'].fillna('') +' '+ meta['abstract'].fillna('')+' ' + meta['keywords'].fillna('')
temp_cancer = [any([re.search(r'cancer ',i) is not None,re.search(r' cancer',i) is not None,
                    re.search(r'tumor ',i) is not None,re.search(r' tumor',i) is not None,
                    re.search(r'metastasis ',i) is not None, re.search(r' metastasis',i) is not None,
                    re.search(r'tme ',i) is not None,re.search(r' tme',i) is not None,
                                                            ]) for i in combine_text.str.lower()]

meta.involve_cancer = temp_cancer
```