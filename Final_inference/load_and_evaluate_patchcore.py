import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import torch
import PIL

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
import torch
from torchvision import transforms
import wandb

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],"visa": ["patchcore.datasets.visa", "VisADataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)

def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    image_array, image = methods["get_dataloaders_iter"](seed)
    
    patchcore_iter  = methods["get_patchcore_iter"]


    patchcore_iter = patchcore_iter(device)
    n_dataloaders = 1

    PatchCore_list = next(patchcore_iter)
   
    aggregator = {"scores": [], "segmentations": []}
    for i, PatchCore in enumerate(PatchCore_list):
        torch.cuda.empty_cache()
        LOGGER.info(
            "Embedding test data with models ({}/{})".format(
                i + 1, len(PatchCore_list)
            )
        )
        #scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
        #   image_array
        #)
        scores, segmentations = PatchCore.predict(
           image_array
        )
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)

    scores = np.array(aggregator["scores"])
    print(scores)
    # min_scores = scores.min(axis=-1).reshape(-1, 1)
    # max_scores = scores.max(axis=-1).reshape(-1, 1)
    # scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)

    segmentations = np.array(aggregator["segmentations"])
    print('seg', segmentations.shape)
    min_scores = (
        segmentations.reshape(len(segmentations), -1)
        .min(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    max_scores = (
        segmentations.reshape(len(segmentations), -1)
        .max(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    print(max_scores)
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations = np.mean(segmentations, axis=0)

    print(scores)
    patchcore.utils.plot_segmentation_images(
                    results_path,
                    image[0],
                    segmentations)
                
    """
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # Plot Example Images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                patchcore.utils.plot_segmentation_images(
                    results_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            ######## added code completed -----------

           

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))
            del PatchCore_list
            gc.collect()

            

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )
        """

@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", get_patchcore_iter)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=366, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
#@click.option("--uncertainty", default=0, type=int, show_default=True)
def dataset(
    name, image_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    
    def get_dataloaders_iter(seed):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        image = PIL.Image.open(image_path).convert("RGB")
        """
        transform_img_1 = [
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        transform_img_1 = transforms.Compose(transform_img_1)

        image = transform_img_1(image)
    
        return [image], [image]
        """
        
        #if uncertainty == 1:
        print("Uncetainty ")
        transform_img = [
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

        tta_images = []
        tta_samples = 10
        for i in range(tta_samples):
            if i <= 3:
                tta_images.append([transform_img[i](image)])
            else:
                j = 4
                tta_images.append([transform_img[j](image)])
        
        transform_img_1 = [
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

        transform_img_1 = transforms.Compose(transform_img_1)
        image = transform_img_1(image)

        return tta_images, [image]
        """
        if uncertainty == 0:
           
            transform_img_1 = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
             ]
            transform_img_1 = transforms.Compose(transform_img_1)

            image = transform_img_1(image)
        
            return [image], [image] 
        """

    return ("get_dataloaders_iter", get_dataloaders_iter)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
