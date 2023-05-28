import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import cv2

from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
   
  
# Importing Image module from PIL package 
from PIL import Image 

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)
    
    def mask_transform_1(mask):
        new_size = (366,366)
        #new_size = (256,256)
        mask =  mask.resize(new_size, resample=Image.NEAREST)
        width, height, _ = np.array(mask).shape

        # Set the desired output size
        new_size1 = 224

        # Calculate the crop box
        left = (width - new_size1) // 2
        top = (height - new_size1) // 2
        right = (width + new_size1) // 2
        bottom = (height + new_size1) // 2

        # Crop the image
        mask = mask.crop((left, top, right, bottom))

        return np.array(mask)

    def image_transform_1(mask):
        new_size = (366,366)
        #new_size = (256,256)
        mask =  mask.resize(new_size, resample=Image.NEAREST)
        width, height, _ = np.array(mask).shape

        # Set the desired output size
        new_size1 = 224

        # Calculate the crop box
        left = (width - new_size1) // 2
        top = (height - new_size1) // 2
        right = (width + new_size1) // 2
        bottom = (height + new_size1) // 2

        # Crop the image
        mask = mask.crop((left, top, right, bottom))

        return np.array(mask)


    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform_1(image)
        #if not isinstance(image, np.ndarray):
        #    image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                #mask = mask_transform(mask)
                mask = mask_transform_1(mask)
                #if not isinstance(mask, np.ndarray):
                #    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
                #mask = mask.transpose(1,2,0)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        #f, axes = plt.subplots(1, 1)

        cmap = plt.cm.get_cmap('hot')
        heatmap = cmap(segmentation)
        heatmap = np.delete(heatmap, 3, 2)
        #heatmap = np.transpose(heatmap, (2, 0, 1))

        #blended = 0.7 * image.transpose(1,2,0) + 0.3 * heatmap
        #alpha = 0.7
        #blended = cv2.addWeighted(image.transpose(1,2,0), alpha, heatmap, 1-alpha, 0)
        #print("image type", image.dtype)
        #boundary = mark_boundaries(np.array(image.transpose(1, 2, 0)), np.array(rgb2gray(mask.transpose(1,2,0))), color=(0, 0, 1), mode='thick')
        #print("boundary type", boundary.dtype)
        #blended = cv2.addWeighted(image.transpose(1, 2, 0), 0.7, np.uint8(boundary*255), 0.3, 0)

        #img_blended = Image.fromarray(np.uint8(np.array(img) * (1 - heatmap_norm[:, :, np.newaxis]) + boundary_map * heatmap_norm[:, :, np.newaxis]))

        seg_arr  = np.uint8(np.array(heatmap)*255)
        #img_arr = np.array(image.transpose(1, 2, 0))
        img_arr = np.array(image)
        #mask_arr = np.array(rgb2gray(mask.transpose(1,2,0)))
        mask_arr = np.array(np.uint8(rgb2gray(mask)))


        boundary_map = mark_boundaries(img_arr, mask_arr, mode="thick")

        alpha = 0.4
        # Create a copy of the original image and blend with the segmentation array
        blended_img = np.copy(img_arr)

        blended_img = alpha * seg_arr + (1 - alpha) * blended_img

        # Overlay the boundary map on the blended image
        blended_img_with_boundary = mark_boundaries(np.uint8(blended_img), mask_arr, color=(0,0,1), mode='thick')

        # Convert the resulting numpy array to a PIL image and save
        result_img = PIL.Image.fromarray((blended_img_with_boundary * 255).astype(np.uint8))

        for ax in axes.flatten():
            ax.axis('off')
        axes[0].imshow(image)
        #axes[1].imshow(mask.transpose(1, 2, 0))
        axes[1].imshow(mask)
        axes[2].imshow(result_img)
        ##axes[0].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

        
        #plt.imshow(segmentation, cmap='hot', interpolation='nearest')

        savename = image_path.split("/")
        savename = "__".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)

        plt.imshow(heatmap)
        plt.axis('off')
        plt.savefig(savename, bbox_inches='tight', pad_inches=0.)  

        #print(savename)
        #cv2.imwrite(savename, segmentation)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
