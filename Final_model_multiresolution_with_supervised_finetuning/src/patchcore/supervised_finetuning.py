import torch 
import cv2
from PIL import Image
import albumentations as A

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm


from torch.utils.data import Dataset

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss
from torch.utils.data import DataLoader
import wandb


def model_training(train_df,valid_df,dataset_class, run_save_path):

    DEVICE = 'cuda'

    EPOCHS = 100  # 150
    LR = 0.0001   # 0.003
    IMAGE_SIZE = 224
    BATCH_SIZE = 2

    ENCODER = 'timm-efficientnet-b0'
    WEIGHTS = 'imagenet'

    def get_train_augs():
        return A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

    def get_valid_augs():
        return A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE)
    ]) 
    
    
    class SegmentationDataset(Dataset):
        def __init__(self,df,augmentations):
            self.df = df
            self.augmentations = augmentations
            #self.selected_maps = selected_maps

        def __len__(self):
            return len(self.df)

        def __getitem__(self,idx):
            row = self.df.iloc[idx]


            mask_path = row.Masks



            image = row.Images
            image = np.array(image)


            if mask_path is not None:
                mask = Image.open(mask_path)
                mask = np.array(mask)/255.0
            else: 
                mask = np.zeros_like(image)

            mask_transform = A.Compose([
                        A.Resize(366,366, interpolation=cv2.INTER_NEAREST),
                        A.CenterCrop(224,224)
                        ])
            mask = mask_transform(image=mask)['image']


            if self.augmentations:
                data = self.augmentations(image=image,mask=mask)
                image = data['image']
                mask = data['mask']



            
            image = torch.Tensor(image)
            mask = torch.Tensor(mask)
            return image,mask

    trainset = SegmentationDataset(train_df, get_train_augs())
    validset = SegmentationDataset(valid_df, get_valid_augs())
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE)



    class SegmentationModel(nn.Module):
        def __init__(self):
            super(SegmentationModel,self).__init__()

            self.arc = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights = WEIGHTS,
                in_channels = 1,
                classes = 1,
                activation = None
            )

        def forward(self,images,masks=None):
            logits = self.arc(images)

            if masks!=None:
                loss1 = DiceLoss(mode='binary')(logits,masks)
                loss2 = nn.BCEWithLogitsLoss()(logits,masks)
                loss = FocalLoss(mode='binary')(logits,masks)
                return logits, 0.5*(loss1 + loss2)
            #return logits, loss
            
            return logits 


    model = SegmentationModel()
    model.to(DEVICE)

    def train_fn(data_loader,model,optimizer):
        model.train()
        total_loss=0.0

        for images,masks in tqdm(data_loader):

            images = images.unsqueeze(1).to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)

            #print(images.shape)
            optimizer.zero_grad()
            logits,loss = model(images,masks)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        return total_loss/len(data_loader) 


    def eval_fn(data_loader,model):
        model.eval()
        total_loss=0.0

        with torch.no_grad():
            for images,masks in tqdm(data_loader):

                images = images.unsqueeze(1).to(DEVICE)
                masks = masks.unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                logits,loss = model(images,masks)

                total_loss+=loss.item()

            return total_loss/len(data_loader)  


    optimizer = torch.optim.Adam(model.parameters(), lr= LR)

    best_valid_loss = np.Inf
    wandb.init(project="VISA_paper_result", entity="fyp_anomaly_detection")

    for i in range(EPOCHS):

        train_loss = train_fn(trainloader,model,optimizer)
        valid_loss = eval_fn(validloader,model)


        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(),  run_save_path + f'/best_model_{dataset_class}.pt')
            print('SAVED-MODEL')
            best_valid_loss  =valid_loss

        print(f'Epoch: {i+1} Train_loss: {train_loss} Valid_loss:{valid_loss}')
        wandb.log({
            'train loss' : train_loss,
            'val loss' : valid_loss,
            'epoch' : i + 1
        })

    wandb.log({
            "learning rate" : LR,
            "total epochs" : EPOCHS,
            "Loss scale" : 0.5
        })