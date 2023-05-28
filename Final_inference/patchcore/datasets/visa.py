import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

_CLASSNAMES = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "macaroni1",
    "macaroni2",
    "capsules",
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for VisA.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            transforms.Compose([
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]),
            transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]),
            transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]),
            transforms.Compose([ transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            ]

        

        self.transform_img_1 = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img_1 = transforms.Compose(self.transform_img_1)

        self.transform_img_2 = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img_2 = transforms.Compose(self.transform_img_2)
        """
        self.transform_mask = [
            transforms.Resize(resize, interpolation=Image.NEAREST),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask) """

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        #classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        classname, anomaly, image_path= self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        #image = self.transform_img(image)

        tta_images = []
        tta_samples = 10
        for i in range(tta_samples):
            if i <= 3:
                tta_images.append([self.transform_img[i](image)])
            else:
                j = 4
                tta_images.append([self.transform_img[j](image)])

       # if self.split == DatasetSplit.TEST and mask_path is not None:
       #     mask = PIL.Image.open(mask_path)
        #    mask = self.transform_mask(mask)
        #else: 
        #mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": tta_images,
            #"mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):

        imgpaths_per_class = {}
        #maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            #maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)
            print(anomaly_types)

            imgpaths_per_class[classname] = {}
            #maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                #print(anomaly)
                anomaly_path = os.path.join(classpath, anomaly)
                print(anomaly_path)
                print(os.listdir(anomaly_path))
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                """
                if anomaly == 'bad' and self.split == DatasetSplit.TEST :
                    anomaly_path = os.path.join(classpath, anomaly)  ## g b 
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]
                if anomaly == 'good' and self.split == DatasetSplit.TRAIN:
                    anomaly_path = os.path.join(classpath, anomaly)  ## g b 
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]
                

                if self.train_val_split < 1.0:
                    print('hi')
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        print('hi1')
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        print('hi2')
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]
                
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else: """
                #maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    
                    data_tuple = [classname, anomaly, image_path]
                    #if self.split == DatasetSplit.TEST and anomaly != "good":
                    #    data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    #else:
                    #data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        print(imgpaths_per_class, data_to_iterate)
        return imgpaths_per_class, data_to_iterate