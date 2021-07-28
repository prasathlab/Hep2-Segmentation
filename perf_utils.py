import numpy as np
import matplotlib.pyplot as plt
import pdb
import wandb
import os
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, jaccard_score, f1_score)
from sklearn.metrics import precision_recall_fscore_support, classification_report


def log_train_history(hist_dict, train_dir, style="seaborn-whitegrid"):

    plt.style.use(style)
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 6), dpi=200)
    x = [int(i) for i in range(1, len(hist_dict["loss"]) + 1)]
    ax1.plot(x, hist_dict["loss"], color='red', linestyle='-', linewidth=1, marker='o', markersize=5, label='Training')
    ax1.plot(x, hist_dict["val_loss"], color='green', linestyle='-', linewidth=1, marker='o', markersize=5, label='Validation')
    ax1.tick_params(axis="y", labelsize=16)
    #ax1.tick_params(axis="x", labelsize=12, rotation=90)
    ax1.tick_params(axis="x", labelsize=12)
    #ax1.set_title("Model Loss", size=20)
    ax1.set_ylabel("Loss", size=20)
    ax1.set_xlabel("Epoch", size=20)
    ax1.legend(loc="upper left", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xlim(0, )
    ax1.set_ylim(0, )
    fig.savefig(os.path.join(train_dir, "Loss.png"))
    plt.close()

    fig, ax2 = plt.subplots(1, 1, figsize=(18, 6), dpi=200)
    #acc_fig = plt.figure()
    ax2.plot(x, hist_dict["accuracy"], color='red', linestyle='-', linewidth=1, marker='o', markersize=5, label='Training')
    ax2.plot(x, hist_dict["val_accuracy"], color='green', linestyle='-', linewidth=1, marker='o', markersize=5, label='Validation')
    ax2.tick_params(axis="y", labelsize=16)
    # ax2.tick_params(axis="x", labelsize=12, rotation=90)
    ax2.tick_params(axis="x", labelsize=12)
    #ax2.set_title("Model Accuracy", size=20)
    ax2.set_ylabel("Accuracy", size=20)
    ax2.set_xlabel("Epoch", size=20)
    ax2.legend(loc="upper left", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xlim(0, )
    ax2.set_ylim(0, 1.1)
    fig.savefig(os.path.join(train_dir, "Accuracy.png"))
    plt.close()

    fig, ax3 = plt.subplots(1, 1, figsize=(18, 6), dpi=200)
    ax3.tick_params(axis="y", labelsize=16)
    ax3.tick_params(axis="x", labelsize=12)
    ax3.set_ylabel("Learning Rate", size=20)
    ax3.set_xlabel("Epoch", size=20)
    ax3.set_xticks(x)
    ax3.set_yscale('log')
    ax3.set_xlim(0, )
    ax3.set_ylim(1e-7, 1e-1)
    ax3.plot(x, hist_dict["lr"], color='green', linestyle='-', linewidth=1, marker='o', markersize=5)
    fig.savefig(os.path.join(train_dir, "Learning_Rate.png"))
    plt.close()



def compute_perf_metrics(test_masks, test_predictions, labels, target_names, threshold_confusion=0.5):
    res_dict = {}
    res_dict["Threshold"] = threshold_confusion

    y_true = np.squeeze(test_masks).flatten()
    y_scores = np.squeeze(test_predictions).flatten()
    y_pred = np.where(y_scores >= threshold_confusion, 1, 0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    #print(conf_matrix)
    report_dict = classification_report(y_true, y_pred,
                                        labels=labels,
                                        target_names=target_names,
                                        digits=4,
                                        output_dict=True,
                                        zero_division='warn'
                                        )
    res_dict["Accuracy"] = report_dict["accuracy"]
    #In binary classification, recall of the negative class is “specificity”.
    res_dict["Specificity"] = report_dict[target_names[0]]["recall"]
    # In binary classification, recall of the positive class is also known as “sensitivity”;
    res_dict["Sensitivity"] = report_dict[target_names[0]]["recall"]
    #We care only about Precision of Positive Class
    res_dict["Precision"] = report_dict[target_names[1]]["precision"]
    #F1 score
    res_dict["F1_score"] = report_dict[target_names[1]]["f1-score"]
    jaccard_index = jaccard_score(y_true, y_pred)
    res_dict["Jaccard"] = jaccard_index
    dice_score = dice_coeff(y_true, y_pred)
    res_dict["Dice"] = dice_score
    return res_dict, report_dict

def dice_coeff(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    union = union.astype("float64")
    if union == 0:
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    intersection = intersection.astype("float64")
    dice_score = 2. * (intersection / union)
    return dice_score

def log_images(original_imgs, masks, predictions, scores, class_labels, img_idxs, ex, thereshold=0.5):
    for idx, img_idx in enumerate(img_idxs):
        original_image = original_imgs[img_idx]
        ground_truth_mask = masks[img_idx]
        prediction_mask = predictions[img_idx]
        prediction_mask = np.where(prediction_mask >= thereshold, 1, 0)
        prediction_mask.astype(dtype="int64")
        wandb.log({f"{ex}_ex_{idx}": wandb.Image(original_image,  caption=f"Jaccard={scores[img_idx]}",
                                                 masks={"predictions": {"mask_data": prediction_mask,
                                                                        "class_labels": class_labels},
                                                        "ground_truth": {"mask_data": ground_truth_mask,
                                                                         "class_labels": class_labels}
                                                        }
                                                 )
                   }
                  )

def get_confusion_matrix(test_masks, test_predictions, threshold_confusion):
    y_scores = np.squeeze(test_predictions).flatten()
    y_true = np.squeeze(test_masks).flatten()
    print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
    y_pred = np.where(y_scores >= threshold_confusion, 1, 0)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return confusion



def get_vis_img_idxs(test_masks, test_predictions, threshold=None):
    test_masks = np.squeeze(test_masks)
    test_predictions = np.squeeze(test_predictions)
    if threshold is not None:
        test_predictions = np.where(test_predictions>=threshold, 1., 0.)
    test_scores = np.multiply(test_masks, test_predictions).mean(axis=(1,2))
    sorted_idxs = np.argsort(test_scores)
    return test_scores, sorted_idxs

def make_roc_curve(y_true, y_scores, test_dir, style="seaborn-whitegrid"):
    y_scores = np.squeeze(y_scores).flatten()
    y_true = np.squeeze(y_true).flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    plt.style.use(style)
    roc_fig, ax1 = plt.subplots(1, 1, figsize=(12, 12), dpi=200)

    ax1.plot(fpr, tpr, color='green', linestyle='-', linewidth=1, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    #ax1.set_title('ROC curve')
    ax1.set_xlabel("FPR (False Positive Rate)", size=20)
    ax1.set_ylabel("TPR (True Positive Rate)", size=20)
    ax1.legend(loc="lower right", fontsize=16)
    roc_fig.savefig(os.path.join(test_dir, "ROC.png"))
    return roc_fig, AUC_ROC

def make_pr_curve(y_true, y_scores, test_dir, style="seaborn-whitegrid"):
    y_scores = np.squeeze(y_scores).flatten()
    y_true = np.squeeze(y_true).flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    plt.style.use(style)
    prec_rec_fig, ax1 = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
    ax1.plot(recall, precision, color='green', linestyle='-', linewidth=1,
             label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    #ax1.set_title('Precision - Recall curve')
    ax1.set_xlabel("Recall", size=20)
    ax1.set_ylabel("Precision", size=20)
    ax1.legend(loc="lower right", fontsize=16)
    prec_rec_fig.savefig(os.path.join(test_dir, "Precision_recall.png"))
    return prec_rec_fig, AUC_prec_rec













    #loss_fig = plt.figure()
    #plt.plot(hist_dict["loss"], color='red', linestyle='-', linewidth=1, marker='o', markersize=5, label='Training')
    #plt.plot(hist_dict["val_loss"], color='green', linestyle='-', linewidth=1, marker='o', markersize=5, label='Validation')
    #plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    #plt.ylim(bottom=0)
    #plt.legend(loc="upper right")
    #plt.savefig(os.path.join(train_dir, "Loss.png"))


# plt.plot(hist_dict["accuracy"], color='red', linestyle='-', linewidth=1, marker='o', markersize=5, label='Training')
    #
    # plt.plot(hist_dict["val_accuracy"], color='green', linestyle='-', linewidth=1, marker='o', markersize=5,
    #          label='Validation')
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.ylim(bottom=0, top=1.)
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(train_dir, "Accuracy.png"))
    # plt.close()


# plt.xlabel("Epochs")
# plt.ylabel("Learning Rate")
# plt.ylim(bottom=0)
# plt.legend(loc="upper right")
# plt.savefig(os.path.join(train_dir, "Learning_Rate.png"))
