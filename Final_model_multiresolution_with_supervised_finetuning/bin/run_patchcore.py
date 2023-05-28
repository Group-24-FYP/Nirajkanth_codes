import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import wandb

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

import patchcore.supervised_finetuning as sf
import patchcore.supervised_finetuning_inference_modified as infer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"], "visa": ["patchcore.datasets.visa", "VisADataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--kshot", default=27, type=int, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--log_online", is_flag=True)
#@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
#@click.option("--patchsize", type=int, default=3)
#@click.option("--resize1", default=366, type=int, show_default=True)
#@click.option("--imagesize1", default=224, type=int, show_default=True)
#@click.option("--resize2", default=256, type=int, show_default=True)
#@click.option("--imagesize2", default=224, type=int, show_default=True)
#@click.option("--resize3", default=400, type=int, show_default=True)
#@click.option("--imagesize3", default=224, type=int, show_default=True)
@click.option("--resize1", default=366, type=int, show_default=True)
@click.option("--imagesize1", default=224, type=int, show_default=True)
@click.option("--resize2", default=256, type=int, show_default=True)
@click.option("--imagesize2", default=224, type=int, show_default=True)
@click.option("--resize3", default=500, type=int, show_default=True)
@click.option("--imagesize3", default=224, type=int, show_default=True)



def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_online,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    resize1,resize2, resize3,
    imagesize1, imagesize2, imagesize3,
    kshot
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    #resize = [resize1,resize2, resize3]
    #imagesize = [imagesize1, imagesize2, imagesize3]
    #list_of_dataloaders = methods["get_dataloaders"](seed, resize, imagesize)

    list_of_dataloaders1 =  methods["get_dataloaders"](seed, resize1, imagesize1)
    list_of_dataloaders2 =  methods["get_dataloaders"](seed, resize2, imagesize2)
    list_of_dataloaders3 =  methods["get_dataloaders"](seed, resize3, imagesize3)

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

    for j, dataloaders in enumerate(list_of_dataloaders1):
    #i = 0
    #for dataloader1, dataloader2, dataloader3 in zip(list_of_dataloaders1, list_of_dataloaders2, list_of_dataloaders3):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                j + 1,
                len(list_of_dataloaders1),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            #imagesize = dataloaders["training"].dataset.imagesize
            imagesize = [imagesize1, imagesize2, imagesize3]
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                #PatchCore.fit(dataloaders["training"])
                if i == 0:
                    PatchCore.fit(list_of_dataloaders1[j]["training"])
                elif i == 1:
                    PatchCore.fit(list_of_dataloaders2[j]["training"])
                elif i == 2:
                    PatchCore.fit(list_of_dataloaders3[j]["training"])

            torch.cuda.empty_cache()
            #aggregator = {"scores": {'score1':[],'score2':[],'score3':[]}, 
            #            "segmentations": {'seg_score1': [], 'seg_score2': [], 'seg_score3': []}}
            aggregator = {"scores": [], 
                        "segmentations": {'seg_score1': [], 'seg_score2': [], 'seg_score3': []}}

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                #print('p', PatchCore)
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                if i == 0:
                    scores, segmentations, labels_gt, masks_gt = PatchCore.predict(list_of_dataloaders1[j]["testing"])
                    #print('sc1', np.array(scores).shape)
                    #print('seg sc1', np.array(segmentations).shape)
                    #aggregator["scores"]['score1'].append(scores)
                    #aggregator["segmentations"]['seg_score1'].append(segmentations)
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"]['seg_score1'].extend(segmentations)

                    break
                """
                elif i == 1:
                    scores, segmentations, labels_gt, masks_gt2 = PatchCore.predict(list_of_dataloaders2[j]["testing"])
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"]['seg_score2'].extend(segmentations)
                elif i == 2:
                    scores, segmentations, labels_gt, masks_gt3 = PatchCore.predict(list_of_dataloaders3[j]["testing"])
                    #aggregator["scores"]['score3'].append(scores)
                    #aggregator["segmentations"]['seg_score3'].append(segmentations)
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"]['seg_score3'].extend(segmentations)
                """



            
            #print(aggregator['scores'])
            scores = np.array(aggregator["scores"])
            ###scores = np.max(scores, axis=1)
            #scores = scores.squeeze()
            #print('score_', scores.shape, scores)

            #scores = np.max(scores, axis=1).reshape(-1,1)
            print('score_', scores.shape)
            s1 = np.expand_dims(np.array(aggregator["segmentations"]['seg_score1']), axis= 0)
            #s2 = np.expand_dims(np.array(aggregator["segmentations"]['seg_score2']), axis= 0)
            #s3 = np.expand_dims(np.array(aggregator["segmentations"]['seg_score3']), axis= 0)
            #print(s1.shape, s2.shape, s3.shape)
            #segmentations = np.concatenate((s1, s2, s3), axis=0)
            segmentations = s1
            print('segmentation', segmentations.shape)
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            print('min_scores', min_scores , 'max_scores' , max_scores)
            scores = (scores - min_scores) / (max_scores - min_scores)
            #print('score_1', scores)
            scores = np.mean(scores, axis=0)
            print('score_shape after mean', scores.shape)
            ##scores = scores.reshape(-1,1)
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
            #print('min_scores', min_scores.shape)
            #print('min_scores', min_scores.shape)
            ##print('min_scores', min_scores)
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            
            print('segmentation before mean', segmentations.shape)
            segmentations = np.mean(segmentations, axis=0)
            print('segmentation after mean', segmentations.shape)
            ###shape = segmentations.shape
            ###segmentations = (1./(1 + np.exp(segmentations))).reshape(shape)
            ###print(segmentations)

            ##print('segmentation after mean', segmentations.shape)
            ##print('segmentation after mean', segmentations)
            #print(list_of_dataloaders1[j]['testing'].name)
            anomaly_labels = [
                x[1] != "good" for x in list_of_dataloaders1[j]["testing"].dataset.data_to_iterate
            ]
            #print('anomaly labels', anomaly_labels)

            # (Optional) Plot example images.

            ### select few segmentation map randomly 

            np.random.seed(42)
            print("kshot :", kshot)
            index_array = np.random.choice(segmentations.shape[0], size=kshot, replace=False)

            selected_maps = segmentations[index_array]

            # Print the shape of the selected maps
            print('selected maps', selected_maps.shape)

            mask_paths = [ x[3] for x in list_of_dataloaders1[j]["testing"].dataset.data_to_iterate]
            selected_mask_paths = [mask_paths[i] for i in index_array]
            print(selected_mask_paths)

            data = {'Images': [inp.tolist() for inp in selected_maps], 'Masks': selected_mask_paths}

            # create a pandas dataframe from the dictionary
            df = pd.DataFrame(data)
            train_df, valid_df = train_test_split(df, test_size =0.4, random_state=42)
            

            sf.model_training(train_df,valid_df,dataloaders["training"].name, run_save_path)
            segmentations_final = infer.sf_infer(segmentations,mask_paths,dataloaders["training"].name,list_of_dataloaders1[j]["testing"], run_save_path)

            if save_segmentation_images:
                image_paths = [
                    x[2] for x in list_of_dataloaders1[j]["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in list_of_dataloaders1[j]["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        list_of_dataloaders1[j]["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        list_of_dataloaders1[j]["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = list_of_dataloaders1[j]["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return list_of_dataloaders1[j]["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                image_save_path_final = os.path.join(
                    run_save_path, "segmentation_images", dataset_name + "_final"
                )
                os.makedirs(image_save_path, exist_ok=True)
                os.makedirs(image_save_path_final, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

                patchcore.utils.plot_segmentation_images(
                    image_save_path_final,
                    image_paths,
                    segmentations_final,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics_rp.")
            """
            s = np.array(scores).reshape(1,-1)
            def sigmoid(x):
                return (1./(1+ np.exp(-x))).reshape(-1,1)

            # Example usage
            scores = sigmoid(s)"""

            
            
            print('scores',scores.shape) 
            ##print(scores.shape)
            print('anomaly_labels shape ', np.array(anomaly_labels).shape)
            #a1 = patchcore.metrics.compute_imagewise_retrieval_metrics(
            #    scores, anomaly_labels
            #)["auroc"]  
            a1 = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            auroc = a1['auroc']
            ap = a1['ap']

            # Compute PRO score & PW Auroc for all images
            #maskgt = np.concatenate((np.array(maskgt1),np.array(maskgt2), np.array(maskgt3) ), axis=0)
            
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations_final, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"] 
            full_pixel_ap = pixel_scores["ap"] 

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            #print(len(masks_gt), 
            print(np.array(masks_gt).shape)
            masks_gt = np.array(masks_gt).squeeze()
            
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            #print(sel_idxs)
            mask_paths = [ x[3] for x in list_of_dataloaders1[j]["testing"].dataset.data_to_iterate]
            #print(mask_paths)
            sel_mask_paths = [mask_paths[p] for p in range(len(mask_paths)) if p in sel_idxs]
            rem_paths = [mask_paths[p] for p in range(len(mask_paths)) if p not in sel_mask_paths ]
            ###print(len(mask_paths), len(sel_mask_paths))
            ###print(len(rem_paths), rem_paths)
            #print(sel_mask_paths)
            #print(segmentations[2])
            #print(masks_gt[2])
            #print(np.unique(masks_gt[2]))
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations_final[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs], 1, sel_mask_paths) 
            anomaly_pixel_auroc = pixel_scores["auroc"] 
            anomaly_pixel_ap = pixel_scores["ap"] 
            
            ## added code  -------------------------------------------------
            
            aupr1 = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            ) 
            aupr = a1['aupr']
            ####auroc, aupr = 0.5, 0.5
            ################
           
            #print('scores', s)
            #scores = (s - s.min())/(s.max() + s.min())
            #print(scores.shape)
            
            img_th_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores , anomaly_labels
                    )["th_auroc"]
            img_th_aupr = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores , anomaly_labels
                    )["th_aupr"] 

            print('opt_th_aupr', img_th_aupr)
            print('opt_th_auroc', img_th_auroc)       

            pred_aupr = (scores >= img_th_aupr).astype(int)
            pred_auroc = (scores >= img_th_auroc).astype(int)


            #print('scores', scores)

            #print('pred_aupr', pred_aupr)
            #print('pred_auroc', pred_auroc)
            #print('gt', anomaly_labels)
            print('no of anomaly samples', np.array(anomaly_labels).sum())
            print('no of predicted anomaly samples(aupr th)', np.array(pred_aupr).sum())
            #print(np.array(anomaly_labels).astype(int).ravel())
            #print(np.array(anomaly_labels).astype(int).shape)
            #print(np.array(pred_aupr).astype(int).ravel())
            #print(np.array(pred_aupr).astype(int).shape)
            print('no of wrong predictions', (np.array(anomaly_labels).astype(int).ravel() - np.array(pred_aupr).astype(int).ravel()).sum())

            #print('no of predicted anomaly samples(auroc th)', np.array(pred_auroc).sum())
            #print('no of wrong predictions', (np.array(anomaly_labels).astype(int).ravel() - np.array(pred_auroc).astype(int).ravel()).sum())
            
            # Compute PRO score & PW Auroc for all images
            
            pixel_scores_aupr = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations_final, masks_gt
            )
            full_pixel_aupr = pixel_scores_aupr["aupr"] 
            full_pixel_ap_final = pixel_scores_aupr["ap"] 

            # Compute PRO score & PW Auroc only for images with anomalies
            
            sel_idxs_aupr = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs_aupr.append(i) 
            pixel_scores_aupr = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations_final[i] for i in sel_idxs_aupr ], [masks_gt[i] for i in sel_idxs_aupr ]
            )
            anomaly_pixel_aupr = pixel_scores_aupr["aupr"]  
            anomaly_pixel_ap_final = pixel_scores_aupr["ap"]  
        
            ######## added code completed ----------- 

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "patchsize"   : 3,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "instance_aupr": aupr,         ###############-----
                    "full_pixel_aupr": full_pixel_aupr,          ###############----------
                    "anomaly_pixel_aupr": anomaly_pixel_aupr,       ###########----
                    "instance_ap" : ap,
                    "full_pixel_ap" : full_pixel_ap, 
                     "anomaly_pixel_ap": anomaly_pixel_ap,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    try:
                        LOGGER.info("{0}: {1:3.3f}".format(key, item))
                    except:
                        LOGGER.info("{} : {}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                print('saving')
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    if i == 0:
                        PatchCore.save_to_path(patchcore_save_path, prepend)
                        break
            
            if log_online:
                wandb.init(project="VISA_paper_result", entity="fyp_anomaly_detection")
                wandb.log({
                    "dataset_name": dataset_name,
                    'layers' : '3wres50', 
                    'imagesize' : imagesize,
                    'resize' : [resize1, resize2, resize3],
                    "patchsize"   : 3,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "instance_aupr": aupr,        
                    "full_pixel_aupr": full_pixel_aupr,         
                    "anomaly_pixel_aupr": anomaly_pixel_aupr,  
                    "instance_ap" : ap,
                    "full_pixel_ap" : full_pixel_ap, 
                    "anomaly_pixel_ap": anomaly_pixel_ap,
                })

            #print(result_collect)


        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    print(backbone_names)

    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]
    print(layers_to_extract_from)

    
    def get_patchcore(input_shape_array, sampler, device):
        loaded_patchcores = []
        i = 0
        print( backbone_names, layers_to_extract_from_coll)
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
            print('input_shape_array', input_shape_array)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape= [3,input_shape_array[i], input_shape_array[i]],
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            print(f'patchcore {backbone_name} is loaded')
            loaded_patchcores.append(patchcore_instance)
            i += 1
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
#@click.option("--resize1", default=366, type=int, show_default=True)
#@click.option("--imagesize1", default=224, type=int, show_default=True)
#@click.option("--resize2", default=256, type=int, show_default=True)
#@click.option("--imagesize2", default=224, type=int, show_default=True)
#@click.option("--resize3", default=400, type=int, show_default=True)
#@click.option("--imagesize3", default=224, type=int, show_default=True)
@click.option("--resize1", default=366, type=int, show_default=True)
@click.option("--imagesize1", default=224, type=int, show_default=True)
@click.option("--resize2", default=256, type=int, show_default=True)
@click.option("--imagesize2", default=224, type=int, show_default=True)
@click.option("--resize3", default=500, type=int, show_default=True)
@click.option("--imagesize3", default=224, type=int, show_default=True)

@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize1,resize2, resize3,
    imagesize1, imagesize2, imagesize3,
    num_workers,
    augment,
    ):
    
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    #resize = [resize1,resize2, resize3]
    #imagesize = [imagesize1, imagesize2, imagesize3]
    def get_dataloaders(seed, resize, imagesize):
        dataloaders = []
        print('get data loader resize , imagesize', resize, imagesize)
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )
            
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

           

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
