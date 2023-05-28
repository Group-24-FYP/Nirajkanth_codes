import cv2
import torch 
import numpy as np 

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss
import os

import csv
import albumentations as A
from sklearn import metrics
import sys

from PIL import Image
from torchvision import transforms



def sf_infer(org_heatmap, gt_masks_paths, dataset_class):
    DEVICE = 'cuda'



    ENCODER = 'timm-efficientnet-b0'
    #ENCODER = 'resnet50'
    WEIGHTS = 'imagenet'

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
                return logits, loss1 + loss2
            #return logits, loss
            
            return logits 


    def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
        """
        Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
        and ground truth segmentation masks.

        Args:
            anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                    generated segmentation masks.
            ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                                predefined ground truth segmentation masks
        """
        if isinstance(anomaly_segmentations, list):
            anomaly_segmentations = np.stack(anomaly_segmentations)
        if isinstance(ground_truth_masks, list):
            ground_truth_masks = np.stack(ground_truth_masks)
        
        print("anomaly_segmentations", anomaly_segmentations.shape)
        print("ground_truth_masks", ground_truth_masks.shape)

        flat_anomaly_segmentations = anomaly_segmentations.ravel()
        flat_ground_truth_masks = ground_truth_masks.ravel()
        print("flat_anomaly_segmentations", len(flat_anomaly_segmentations))
        print("flat_ground_truth_masks", len(flat_ground_truth_masks))

        fpr, tpr, thresholds = metrics.roc_curve(
            flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
        )
        auroc = metrics.roc_auc_score(
            flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
        )


        precision, recall, thresholds = metrics.precision_recall_curve(
            flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
        )

        au_pr = metrics.auc(recall, precision)

        F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        optimal_threshold = thresholds[np.argmax(F1_scores)]
        predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
        fpr_optim = np.mean(predictions > flat_ground_truth_masks)
        fnr_optim = np.mean(predictions < flat_ground_truth_masks)

        return {
            "auroc": auroc,
            "fpr": fpr,
            "tpr": tpr,
            "optimal_threshold": optimal_threshold,
            "optimal_fpr": fpr_optim,
            "optimal_fnr": fnr_optim,
            "aupr" : au_pr,
        }


    model = SegmentationModel()
    model.to(DEVICE)

    #model.load_state_dict(torch.load("/home/mayooran/mugunthan/patchcore-inspection-main/PATCH_CORE/Supervised_fine_tuning/best_model_tiff.pt"))
    model.load_state_dict(torch.load("/home/mayooran/niraj/patchcore_main_final/patchcore-inspection/best_model.pt"))


    def predict(image):
        
        img_name = f"/home/mayooran/mugunthan/patchcore-inspection-main/PATCH_CORE/Supervised_fine_tuning/PATCH_CORE/patchcore-inspection/results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/segmentation_images/mvtec_grid/{image}"
        
        print(img_name)
        image = Image.open(img_name)###uncommented


        image = np.array(image)

        
        image = torch.Tensor(image)



        logits_mask = model(image.to(DEVICE).unsqueeze(0).unsqueeze(0)) #(C,H,W) -> (1,C,H,W)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask>0.5)*1.0

        #return image
        #return pred_mask.squeeze()
        return pred_mask.squeeze()+image.to(DEVICE)
    # print(pred_mask)


    img_files = os.listdir("/home/mayooran/mugunthan/patchcore-inspection-main/PATCH_CORE/Supervised_fine_tuning/PATCH_CORE/patchcore-inspection/results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/segmentation_images/mvtec_grid/")
    img_files.sort()

    # print(img_files,len(img_files))

    mask_files =[]
    ground_truth_masks_dict = {}
    for path, subdirs, files in os.walk('/home/mayooran/MVTecAD/grid/ground_truth/'):
        for name in files:
            mask_files.append(path.split('/')[-3]+'_test_'+path.split('/')[-1]+'_'+name)

            mask = Image.open(os.path.join(path, name))###uncommented
            mask = np.array(mask)/255.0


            mask_transform = A.Compose([
            A.Resize(400,400, interpolation=cv2.INTER_NEAREST),
            A.CenterCrop(224,224)
            ])
            mask_crop = mask_transform(image=mask)['image']


            ground_truth_masks_dict[path.split('/')[-3]+'_test_'+path.split('/')[-1]+'_'+name]=mask_crop
    mask_files.sort()




    # print(mask_files,len(mask_files))

    # open the file in the write mode
    with open('./train.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(['Images','Masks'])

        for i in range(len(img_files)):
            writer.writerow([img_files[i],mask_files[i]])


    def mask_crop(mask_name):
        mask = cv2.imread(''+mask_name, cv2.IMREAD_GRAYSCALE)/255.0
        mask_transform = A.Compose([
            A.Resize(366,366),
            A.CenterCrop(224,224)
        ])
        mask_crop = mask_transform(image=mask)['image']
        return mask_crop      


    anomaly_segmentations = []



    masks_gt =[]
    for i in img_files:
        print(i)
        pred = predict(i)
        anomaly_segmentations.append(pred.cpu().detach().numpy())
        print(pred.shape)
        #anomaly_segmentations.append(pred)
        masks_gt.append(ground_truth_masks_dict[i[:-5]+'_mask.png'])





    anomaly_segmentations = np.array(anomaly_segmentations)

    print(anomaly_segmentations.shape)
    # Reshape the 3D array to a 2D array
    n, h, w = anomaly_segmentations.shape
    anomaly_segmentations_flat = anomaly_segmentations.reshape(n, h * w)



    output = compute_pixelwise_retrieval_metrics(anomaly_segmentations, masks_gt)
    print(output)
    return anomaly_segmentations
