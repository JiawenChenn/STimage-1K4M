# this code is adopted from CLIP code: https://github.com/openai/CLIP/issues/83
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from transformers import CLIPVisionModel, AutoProcessor, CLIPVisionModelWithProjection
now = datetime.now()

import sys
sys.path.append('.')
from model_st import gene_expression_encoder, ST_contrastive_model, image_gene_data
from model_st import convert_models_to_fp32, convert_weights_tofp16

############# hyperparameters #############
BATCH_SIZE = 1024
EPOCH=15
lr = 5e-5
betas = (0.9,0.98)
eps = 1e-6
weight_decay = 0.2
start_time = now.strftime("%m-%d-%Y-%H:%M:%S")

pre_trained_model = 'openai/clip-vit-base-patch32'
cache_dir = './clip_cache'

gene_type = 'overlap_hvg'
tissue_type = 'Human_Brain_Maynard_02082021_Visium'

save_checkpoint_path = './models/'+pre_trained_model.replace("/","_")+"/"+tissue_type+"_"+start_time+'/'
os.makedirs(save_checkpoint_path,exist_ok=True)
############################################

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.


vision_model = CLIPVisionModelWithProjection.from_pretrained(pre_trained_model,cache_dir=cache_dir)
image_preprocesser = CLIPProcessor.from_pretrained(pre_trained_model,cache_dir=cache_dir).image_processor


data_path = f'./all/{tissue_type}_count_{gene_type}.csv'
data = pd.read_csv(data_path,header=None,index_col=0)
if gene_type == 'overlap_hvg':
    data = pd.read_csv(data_path,index_col=0)


with open(f'{save_checkpoint_path}/hyperparameters.txt','w') as f:
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'EPOCH: {EPOCH}\n')
    f.write(f'lr: {lr}\n')
    f.write(f'betas: {betas}\n')
    f.write(f'eps: {eps}\n')
    f.write(f'weight_decay: {weight_decay}\n')
    f.write(f'model_name: {pre_trained_model}\n')
    f.write(f'start_time: {start_time}\n')
    f.write(f'training_size: {data.shape[0]}\n')
    f.write(f'data_path: {data_path}\n')
    f.write(f'gene_type: {gene_type}\n')
    f.write(f'tissue_type: {tissue_type}\n')


list_image_path = ['./crop/'+slide+'.png' for slide in data.index.tolist()]

gene_expression_data = data.to_numpy()

data_all = image_gene_data(list_image_path,gene_expression_data,image_preprocesser)

data_all_dataloader = DataLoader(data_all,batch_size = BATCH_SIZE,shuffle=True) #Define your own dataloader

st_model = ST_contrastive_model(gene_expression_data.shape[1],512,vision_model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(st_model.parameters(), lr=lr,betas=betas,eps=eps,weight_decay=weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new 

#convert_models_to_fp32(st_model)
st_model.to(device)

convert_models_to_fp32(st_model)

for epoch in range(EPOCH):
    for batch in data_all_dataloader :
        optimizer.zero_grad()
        #
        images,gene_exp = batch 
        #
        images= images.to(device)
        gene_exp = gene_exp.to(device)
        #
        logits_per_image, logits_per_gene = st_model(images, gene_exp)
        #
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        #
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_gene,ground_truth))/2
        total_loss.backward()
        print(f'Epoch: {epoch} Loss: {total_loss.item()}')
        #if device == "cpu":
        #   optimizer.step()
        #else: 
        #    convert_models_to_fp32(st_model)
        optimizer.step()
        #    convert_weights_tofp16(st_model)
    torch.save({
            'epoch': epoch,
            'model_state_dict': st_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"{save_checkpoint_path}/model_epoch{epoch}.pt") #just change to your preferred folder/filename

