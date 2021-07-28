
from __future__ import division
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import sys
from models.vanilla_unet import get_vanilla_unet_model
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau, EarlyStopping
from time import time
from keras import callbacks
from keras.utils import plot_model
from keras.optimizers import Adam
import datetime
import pandas as pd
import yaml

import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import psutil
import tensorflow as tf
from train_utils import *
from perf_utils import *
from file_utils import *
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, jaccard_score, f1_score)
from arg_parser import *
from image_utils import *
from collections import OrderedDict
from vis_utils import *
import json
import pdb


class HEP2_app:
    def __init__(self, yaml_filepath, sys_argv=None):
        # best to just setup the directories in the bat file and pass it via a parser.
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = get_parser()
        self.args = parser.parse_args(sys_argv)

        with open(yaml_filepath, 'r') as stream:
            try:
                self.cfg = yaml.load(stream)
                print(self.cfg)
            except yaml.YAMLError as exc:
                print(exc)

        if self.args.train_model:
            self.res_dir = self.args.base_res_dir
            (self.exp_dir,
             self.train_dir,
             self.test_dir,
             self.tb_dir,
             self.mdl_dir) = setup_results_dir(res_dir=self.res_dir,
                                               tb_dir="tb_log",
                                               time_stamp=True, )
            self.final_mdl_dir = os.path.join(self.mdl_dir, "Final_Model")
            os.makedirs(self.final_mdl_dir, exist_ok=True)
            self.vis_dir = os.path.join(self.test_dir, "Visualization")
            os.makedirs(self.vis_dir, exist_ok=True)

        elif self.args.model_weights is not None:
            self.model_weights = self.args.model_weights
            self.test_dir = self.args.test_dir
            self.vis_dir = os.path.join(self.test_dir, "Visualization")
            os.makedirs(self.vis_dir, exist_ok=True)

        else:
            sys.exit("Not enough args")

        ## Save the yaml file
        with open(os.path.join(self.exp_dir, "run_yaml.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

        # HyperParams
        self.train_bs = self.cfg["train"]["batch_size"]
        self.steps_per_epoch = self.cfg["train"]["steps_per_epoch"]
        self.num_epochs = self.cfg["train"]["num_epochs"]
        self.n_channels = self.cfg["train"]["n_channels"]
        self.lr = float(self.cfg["train"]["learning_rate"])
        self.loss = self.cfg["train"]["loss"]
        self.optimizer_name = self.cfg["train"]["optimizer"]

        if "train" in self.cfg:
            self.__load_train_data()
        if "val" in self.cfg:
            self.__load_val_data()
        if "test" in self.cfg:
            self.__load_test_data()
            self.threshold_confusion = float(self.cfg["test"]["threshold_confusion"])

        self.__init_callbacks()

        if self.use_train_generator:
            train_batch, _ = next(self.train_gen)
            self.model = get_vanilla_unet_model(input_shape=train_batch.shape[1:])
        else:
            self.model = get_vanilla_unet_model(input_shape=self.train_imgs.shape[1:])

        self.model.compile(optimizer=Adam(lr=self.lr), loss=self.loss, metrics=['accuracy'])
        self.model.summary()

    def __init_callbacks(self):
        # mdl_file = os.path.join(train_dir, weights_E_{epoch:02d}_VL_{val_loss:.2f}.hdf5)
        mcp_save = ModelCheckpoint(os.path.join(self.mdl_dir, 'ckpt_E_{epoch:02d}_VL_{val_loss:.2f}.hdf5'),
                                   monitor="val_loss", verbose=0, save_best_only=False,
                                   save_weights_only=False, mode="auto", save_freq='epoch',
                                   )
        # Init the tensorboard dir
        tb_obj = TensorBoard(log_dir=self.tb_dir)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=7,
                                           verbose=1,
                                           epsilon=1e-4,
                                           mode='min'
                                           )
        early_stopping = EarlyStopping(monitor="val_accuracy",
                                       min_delta=1e-3,
                                       patience=7,
                                       verbose=1,
                                       mode="auto",
                                       baseline=None,
                                       restore_best_weights=True,
                                       )
        self.cb_list = [mcp_save, reduce_lr_loss, tb_obj, early_stopping]

    def __load_train_data(self):
        self.use_train_generator = self.cfg["train"]["train_generator"]
        if self.use_train_generator and "val" in self.cfg:
            self.img_files = glob.glob(self.cfg["train"]["image_dir"] + "/*.npy")
            self.train_mask_dir = self.cfg["train"]["mask_dir"]
            # if the config file has val key, it implies val data is available.
            # Hence, splitting data into train and val sets is not required.
            self.train_img_files = self.img_files
            self.train_gen = make_generator(self.train_img_files, self.train_mask_dir, self.train_bs)

        else:
            self.train_imgs = np.load(self.cfg["train"]["input_file"])
            self.train_masks = np.load(self.cfg["train"]["mask_file"])

    def __load_val_data(self):
            #Currently, code does not support val generator. But this is a small change and can be done if needed later
            self.val_img_files = glob.glob(self.cfg["val"]["image_dir"] + "/*.npy")
            self.val_mask_dir = self.cfg["val"]["mask_dir"]
            self.val_imgs = np.load(self.cfg["val"]["input_file"])
            self.val_masks = np.load(self.cfg["val"]["mask_file"])

    def __load_test_data(self):
        assert (self.cfg["prepare_data"]["stride_dim"][0] < self.cfg["prepare_data"]["patch_dim"][0])
        if self.cfg["prepare_data"]["extract_patches"]:
            self.test_df = pd.read_csv(self.cfg["prepare_data"]["test_data"], sep="\t", index_col=0)
            self.norm_params_df = pd.read_csv(self.cfg["test"]["normalization_params"], sep="\t", index_col=0)
            self.ch_mean = self.norm_params_df.loc["Mean", :].to_numpy()
            self.ch_std = self.norm_params_df.loc["Std", :].to_numpy()

        #Only during development.
        # temp_df = self.test_df[:30]
        # pdb.set_trace()
        # self.test_df = temp_df



    def train_model(self):

        if self.use_train_generator and "val" in self.cfg:
            history = self.model.fit(self.train_gen,
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.num_epochs,
                                     callbacks=self.cb_list,
                                     validation_data=(self.val_imgs, self.val_masks),
                                     validation_batch_size=self.train_bs,
                                     validation_freq=1,
                                     verbose=1
                                     )
        else:
            history = self.model.fit(self.train_imgs,
                                     self.train_masks,
                                     batch_size=self.train_bs,
                                     epochs=self.num_epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_split=self.cfg["train"]["valid_ratio"],
                                     callbacks=self.cb_list
                                     )
        #Save the final model
        tf.keras.models.save_model(self.model,
                                   os.path.join(self.final_mdl_dir, "best_model.hdf5"),
                                   overwrite=True, include_optimizer=True,
                                   save_format="h5"
                                   )
        # Save the history.history dict. Useful for plotting loss etc.
        hist_dict = history.history
        self.hist_df = pd.DataFrame.from_dict(hist_dict)
        self.hist_df.to_csv(os.path.join(self.train_dir, "train.tsv"), sep="\t")
        log_train_history(hist_dict, self.train_dir)

    def model_predict_by_patches(self):
        patch_height = self.cfg["prepare_data"]["patch_dim"][0]
        patch_width = self.cfg["prepare_data"]["patch_dim"][1]
        stride_height = self.cfg["prepare_data"]["stride_dim"][0]
        stride_width = self.cfg["prepare_data"]["stride_dim"][1]
        self.img_pred_dict = {}
        #Create a Dict: "iamgeName" : [acc, prec, sen, spe, f1_score, jaccard, dice]
        if self.cfg["prepare_data"]["extract_patches"]:
            for idx1, (idx2, row) in enumerate(self.test_df.iterrows()):
                img_path = row["imageNames"]
                mask_path = row["maskNames"]
                img_name = os.path.basename(img_path)
                test_img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
                test_mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
                test_img, test_mask = image_mask_scaling(test_img, test_mask)
                test_img = (test_img - self.ch_mean) / self.ch_std
                if test_img.ndim ==2:
                    test_img = np.expand_dims(test_img, axis=-1)
                if test_img.ndim == 3:
                    test_img = np.expand_dims(test_img, axis=0)
                assert (test_img.ndim == 4)

                orig_height, orig_width = test_img.shape[1:3]
                test_img_overlap = paint_border_overlap(test_img,
                                                         patch_height, patch_width,
                                                         stride_height, stride_width
                                                         )
                if idx1 == 0:
                    print("new full images shape: \n" + str(test_img_overlap.shape))
                new_height, new_width = test_img_overlap.shape[1:3]
                test_img_patches = extract_ordered_overlap(test_img_overlap,
                                                           patch_height, patch_width,
                                                           stride_height, stride_width
                                                           )

                test_pred_patches = self.model.predict(test_img_patches)
                #print(f"predicted images size : {test_pred_patches.shape}")
                test_pred_img_overlap = recompose_overlap(test_pred_patches,
                                                          new_height, new_width,
                                                          stride_height, stride_width
                                                          )

                ## back to original dimensions
                test_pred_img = test_pred_img_overlap[:, 0:orig_height, 0:orig_width, :]
                self.img_pred_dict[img_name] = test_pred_img
                print(f"Finished Prediction for image {idx1+1} of {self.test_df.shape[0]}")

    def evaluate_model(self):

        all_test_preds = []
        all_test_masks = []
        self.full_report_dict = OrderedDict()
        self.all_res_df = pd.DataFrame()
        for idx1, (idx2, row) in enumerate(self.test_df.iterrows()):
            img_path = row["imageNames"]
            mask_path = row["maskNames"]
            img_name = os.path.basename(img_path)
            self.img_pred_dict["imageName"] = img_name
            test_mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
            test_mask = mask_binarization(test_mask)
            all_test_masks.append(test_mask)
            assert(img_name in self.img_pred_dict)
            test_pred = self.img_pred_dict[img_name]
            all_test_preds.append(test_pred)
            res_dict, report_dict = compute_perf_metrics(test_mask, test_pred,
                                            labels=self.cfg["test"]["labels"],
                                            target_names=self.cfg["test"]["target_names"],
                                            threshold_confusion=self.threshold_confusion)
            self.full_report_dict[img_name] = report_dict
            res_df = pd.DataFrame(res_dict, index=[img_name])

            self.all_res_df = pd.concat([self.all_res_df, res_df])
            print(f"Finished Computing Metrics for image {idx1+1} of {self.test_df.shape[0]}")

        self.all_res_df.to_csv(os.path.join(self.test_dir, "image_performances.tsv"), sep="\t")
        with open(os.path.join(self.test_dir, "full_detail_report.json"), 'w') as fout:
            json_dumps_str = json.dumps(self.full_report_dict, indent=4)
            print(json_dumps_str, file=fout)

        summary_res_dict, summary_report_dict = compute_perf_metrics(all_test_masks, all_test_preds,
                                            labels=self.cfg["test"]["labels"],
                                            target_names=self.cfg["test"]["target_names"],
                                            threshold_confusion=self.threshold_confusion)

        with open(os.path.join(self.test_dir, "summary_detail_report.json"), 'w') as fout:
            json_dumps_str = json.dumps(summary_report_dict, indent=4)
            print(json_dumps_str, file=fout)
        self.full_res_df = pd.DataFrame(summary_res_dict, index=[0])
        _, AUC_ROC = make_roc_curve(all_test_masks, all_test_preds, self.test_dir)
        self.full_res_df.loc[0, "AUC_ROC"] = AUC_ROC
        _, AUC_PR = make_pr_curve(all_test_masks, all_test_preds, self.test_dir)
        self.full_res_df.loc[0, "AUC_PR"] = AUC_PR
        self.full_res_df.to_csv(os.path.join(self.test_dir, "summary_performances.tsv"), sep="\t")


    def visualize_model(self):
        n_vis_imgs = self.cfg["test"]["n_vis_imgs"]
        n_rand_imgs =  self.cfg["test"]["n_rand_imgs"]
        sorted_all_res_df = self.all_res_df.sort_values(by=["Jaccard"], ascending=False)
        top_imgs_df = sorted_all_res_df[:n_vis_imgs]
        bad_imgs_df = sorted_all_res_df[-n_vis_imgs:]
        rand_imgs_df = sorted_all_res_df[n_vis_imgs:-n_vis_imgs].sample(n=n_rand_imgs, replace=False)
        #Visualize Top Images
        top_imgs_df = get_imgs_from_names(top_imgs_df, self.test_df)
        do_contour_visualization(top_imgs_df, self.img_pred_dict, self.vis_dir, threshold_confusion=self.threshold_confusion)
        # Visualize Bad Images
        bad_imgs_df = get_imgs_from_names(bad_imgs_df, self.test_df)
        do_contour_visualization(bad_imgs_df, self.img_pred_dict, self.vis_dir, threshold_confusion=self.threshold_confusion)
        #Visualize Randomly selected images
        rand_imgs_df = get_imgs_from_names(rand_imgs_df, self.test_df)
        do_contour_visualization(rand_imgs_df, self.img_pred_dict, self.vis_dir, threshold_confusion=self.threshold_confusion)





















