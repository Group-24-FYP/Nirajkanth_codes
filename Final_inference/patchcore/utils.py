import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from PIL import Image 
import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_array,
    segmentation,
    save_depth=4
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
    #def image_transform(image):
    #print(image_array.shape)
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]
    in_std = np.array(
                    in_std
                ).reshape(-1, 1, 1)
    in_mean = np.array(
                    in_mean
                ).reshape(-1, 1, 1)
    image =  np.clip(
                    (image_array.numpy() * in_std + in_mean) * 255, 0, 255
                ).astype(np.uint8)
    #if mask_paths is None:
    #    mask_paths = ["-1" for _ in range(len(image_paths))]
    #masks_provided = mask_paths[0] != "-1"
    #if anomaly_scores is None:
    #    anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    # for image_path, anomaly_score, segmentation in tqdm.tqdm(
    #     zip(image_paths, anomaly_scores, segmentations),
    #     total=len(image_paths),
    #     desc="Generating Segmentation Images...",
    #     leave=False,
    # ):
        #image = PIL.Image.open(image_path).convert("RGB")
        # image = image_transform_1(image)
        #if not isinstance(image, np.ndarray):
         #   image = image.numpy()

      

    #print(savefolder)
    #savename = image_path.split("/")
    #savename = "_".join(savename[-save_depth:])

    savename = os.path.join(savefolder, '_seg.png')


    cmap = plt.cm.get_cmap('hot')
    #print(segmentation.shape)
    heatmap = np.squeeze(cmap(segmentation.squeeze()))
    #print(heatmap.shape)
    heatmap = np.delete(heatmap, 3, 2)
    
    #print(heatmap.shape)

    seg_arr  = np.uint8(np.array(heatmap)*255)
    #print('im', image.shape)
    img_arr = np.array(image.transpose(1, 2, 0))
    #print('ima', img_arr.shape)
    #img_arr = np.array(image)
    #print('sarr', seg_arr)
    alpha = 0.4
    #print(seg_arr.shape, seg_arr.dtype)
    #print(img_arr.shape, img_arr.dtype)
    blended_img = alpha * seg_arr + (1 - alpha) * img_arr

    #result_img = (blended_img * 255).astype(np.uint8)

    result_img = PIL.Image.fromarray(blended_img.astype(np.uint8))

    # f, axes = plt.subplots(1, 2)
    # #print('image shape', image.shape)
    # #print(image.transpose(1, 2, 0).shape)
    # axes[0].imshow(image.transpose(1, 2, 0))
    # #axes[1].imshow(mask.transpose(1, 2, 0))
    # axes[1].imshow(segmentation.transpose(1,2,0))
    # f.set_size_inches(3 * (2), 3)
    # f.tight_layout()
    # f.savefig(savename)
    # plt.close()

    plt.imshow(result_img)
    plt.axis('off')
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.)


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"):
    
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
        #if row_names is not None:
        #    header = ["Row Names"] + header

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
