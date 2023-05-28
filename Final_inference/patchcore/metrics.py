"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    
    fpr, tpr, th_auroc = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, th_aupr = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    au_pr = metrics.auc(recall, precision)

    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold_aupr = th_aupr[np.argmax(F1_scores)]
    optimal_threshold_auroc = 0.5

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "th_auroc": optimal_threshold_auroc, 'th_aupr': optimal_threshold_aupr, "aupr" : au_pr}
    #return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, i = 0, sp=None):
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
    
    #print("anomaly_segmentations", anomaly_segmentations.shape)
    #print("ground_truth_masks", ground_truth_masks.shape)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()
    #print("flat_anomaly_segmentations", len(flat_anomaly_segmentations))
    #print("flat_ground_truth_masks", len(flat_ground_truth_masks))
    if i == 1:
        for gt, aseg ,p in zip(ground_truth_masks, anomaly_segmentations, sp):
            gt = gt.ravel()
            aseg = aseg.ravel()
            #print(np.array(aseg).shape)
            #print(len(np.unique(np.array(gt))))
            #print(gt)
            #print(np.unique(gt))
            print(np.unique(gt.astype(int)))
            #print(np.unique(np.array(aseg)))
            
            if len(np.unique(np.array(gt.astype(int)))) != 1:
                fpr, tpr, thresholds = metrics.roc_curve(gt.astype(int), aseg)
                auroc = metrics.roc_auc_score(gt.astype(int), aseg)
                precision, recall, thresholds = metrics.precision_recall_curve(gt.astype(int), aseg)
                au_pr = metrics.auc(recall, precision)

                print("auroc", auroc,  "aupr" , au_pr, 'path', p)
                
        

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
