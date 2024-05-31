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


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

def convert_weights_tofp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


class gene_expression_encoder(nn.Module):
    def __init__(self, 
                 n_in,
                 n_out=128):
        super(gene_expression_encoder, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        return x

class ST_contrastive_model(nn.Module):
    def __init__(self, 
                 gene_expression_dim,
                 latent_dim,
                 vision_model):
        super(ST_contrastive_model, self).__init__()
        self.vision_model = vision_model
        self.gene_expression_encoder = gene_expression_encoder(gene_expression_dim, latent_dim)
        self.image_fc = nn.Linear(512, latent_dim)
        torch.nn.init.xavier_uniform_(self.image_fc.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, image, gene_expression):
        image_features = self.vision_model(image).image_embeds
        image_features = self.image_fc(image_features)
        gene_features = self.gene_expression_encoder(gene_expression)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        gene_features = gene_features / gene_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ gene_features.t()
        logits_per_gene = logits_per_image.t()
        return logits_per_image, logits_per_gene


class image_data(Dataset):
    def __init__(self, list_image_path,image_preprocesser):
        self.image_path = list_image_path
        self.image_preprocesser = image_preprocesser
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        image = self.image_preprocesser(Image.open(self.image_path[idx]),return_tensors="pt")['pixel_values'][0] # Image from PIL module
        return image


class image_gene_data(Dataset):
    def __init__(self, list_image_path, gene_expression,image_preprocesser):
        self.image_path = list_image_path
        self.gene_expression = torch.Tensor(gene_expression)
        self.image_preprocesser = image_preprocesser
        if len(self.image_path) != self.gene_expression.shape[0]:
            raise ValueError('The length of image_path and gene_expression should be the same.')
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        image = self.image_preprocesser(Image.open(self.image_path[idx]),return_tensors="pt")['pixel_values'][0] # Image from PIL module
        gene_expression = self.gene_expression[idx,:]
        return image,gene_expression

class image_gene_data_uni(Dataset):
    def __init__(self, list_image_path, gene_expression,image_preprocesser):
        self.image_path = list_image_path
        self.gene_expression = torch.Tensor(gene_expression)
        self.image_preprocesser = image_preprocesser
        if len(self.image_path) != self.gene_expression.shape[0]:
            raise ValueError('The length of image_path and gene_expression should be the same.')
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        image = self.image_preprocesser(Image.open(self.image_path[idx])) # Image from PIL module
        gene_expression = self.gene_expression[idx,:]
        return image,gene_expression
