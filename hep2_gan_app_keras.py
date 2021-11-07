
from __future__ import division
import sys
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import yaml
import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
from train_utils import *
from perf_utils import *
from file_utils import *
from arg_parser import *
from image_utils import *
from collections import OrderedDict
from vis_utils import *
import json
from constants import *
import pdb
from fcn_models.fcn import *
from contextlib import redirect_stdout
from GANs_keras.gan_discriminator_models import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import shutil
import random
from fcn_models.mobilenet import relu6

class HEP2_GAN_Keras_App:
    def __init__(self, yaml_filepath, args, sys_argv=None):
        # best to just setup the directories in the bat file and pass it via a parser.
        self.args = args
        with open(yaml_filepath, 'r') as stream:
            try:
                self.cfg = yaml.load(stream)
                print(self.cfg)
            except yaml.YAMLError as exc:
                print(exc)

        # HyperParams
        self.train_bs = self.cfg["train"]["batch_size"]
        self.steps_per_epoch = self.cfg["train"]["steps_per_epoch"]
        self.num_epochs = self.cfg["train"]["num_epochs"]
        self.lr = float(self.cfg["train"]["learning_rate"])
        self.loss = self.cfg["train"]["loss"]
        self.optimizer_name = self.cfg["train"]["optimizer"]
        self.gen_arch_name = self.cfg["project"]["gen_arch"]
        self.disc_arch_name = self.cfg["project"]["disc_arch"]
        self.height = self.cfg["data"]["height"]
        self.width = self.cfg["data"]["width"]
        self.n_channels = self.cfg["data"]["n_channels"]
        #Currently only below shape is supported.
        self.input_shape = (self.height, self.width, self.n_channels)
        self.pretrained_w = self.cfg["train"]["pretrained_w"]
        self.fine_tune = self.cfg["train"]["fine_tune"]
        self.n_classes = len(self.cfg["data"]["labels"])
        self.real_label = 1
        self.fake_label = 0

        self.prefix = self.gen_arch_name
        if self.pretrained_w:
            self.prefix = self.prefix + "_" + "PT"
            if self.fine_tune:
                self.prefix = self.prefix + "_" + "FT"
            else:
                self.prefix = self.prefix + "_" + "NFT"
        else:
            # This is just to make sure there is never a case of NPT and NFT
            assert(not self.pretrained_w and self.fine_tune)
            self.prefix = self.prefix + "_" + "NPT" + "_" + "FT"

        if self.args.train_model:
            self.res_dir = self.args.base_res_dir
            (self.exp_dir,
             self.train_dir,
             self.test_dir,
             self.tb_dir,
             self.mdl_dir) = setup_results_dir(res_dir=self.res_dir,
                                               tb_dir="tb_log",
                                               time_stamp=True, prefix=self.prefix)
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

        if "train" in self.cfg:
            self.__load_train_data()
        if "val" in self.cfg:
            self.__load_val_data()
        if "test" in self.cfg:
            self.__load_test_data()
            self.threshold_confusion = float(self.cfg["test"]["threshold_confusion"])

        self.__init_callbacks()

        self.__load_model__()
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        with open(os.path.join(self.train_dir, f'Generator_{self.generator_model.model_name}_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.generator_model.summary()

        with open(os.path.join(self.train_dir, f'Discriminator_{self.discriminator_model.model_name}_summary.txt'),
                  'w') as f:
            with redirect_stdout(f):
                self.discriminator_model.summary()


    def __get_generator_model__(self):
        assert (self.gen_arch_name in Supported_Archs)
        # height, width and channels must be supplied by yaml
        if self.gen_arch_name in ["Vanilla_U-Net", "Res_U-Net", "Dense_U-Net"]:
            # This if condition will be deprecated later
            self.generator_model = Supported_Archs[self.gen_arch_name](self.input_shape)
        else:
            backbone = any(name in self.gen_arch_name for name in Supported_Backbones)
            if backbone:
                self.generator_model = Supported_Archs[self.gen_arch_name](n_classes=self.n_classes,
                                                                           input_height=self.height,
                                                                           input_width=self.width,
                                                                           channels=self.n_channels,
                                                                           pretrained_w=self.pretrained_w,
                                                                           fine_tune=self.fine_tune
                                                                           )
            else:
                print("Here")
                self.generator_model = Supported_Archs[self.gen_arch_name](n_classes=self.n_classes,
                                                                           input_height=self.height,
                                                                           input_width=self.width,
                                                                           channels=self.n_channels,
                                                                           )
        return self.generator_model

    def __make_composite_model__(self):
        #TODO: ADD Identity Loss. i.e. if a mask is input to generator it must return the mask without change.
        # make weights in the discriminator not trainable
        self.discriminator_model.trainable = False
        # discriminator element
        input_gen = Input(shape=(self.height, self.width, self.n_channels))

        # Generated image to be input to d_model_1 (adversarial loss)
        #generate mask from image. Eventually this should get close to real mask
        generated_mask = self.generator_model(input_gen)
        #output_d: is it real or fake
        output_d = self.discriminator_model(generated_mask)
        self.composite_gan_model = Model([input_gen], [output_d, generated_mask])

        # compile model
        self.composite_gan_model.compile(loss=['binary_crossentropy', 'mae'],
                                         optimizer=Adam(lr=0.0002, beta_1=0.5),
                                         loss_weights=[1, 100],
                                         #metrics=['accuracy']
                                         )
        return self.composite_gan_model


    def __load_model__(self):
        self.discriminator_model = Supported_Archs[self.disc_arch_name](image_shape=(self.height, self.width, self.n_channels))
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        self.generator_model = self.__get_generator_model__()
        self.composite_gan_model = self.__make_composite_model__()



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

            if self.cfg["train"]["augment"]:
                self.aug_img_files = glob.glob(self.cfg["train"]["aug_image_dir"] + "/*.npy")
                self.aug_mask_dir = self.cfg["train"]["aug_mask_dir"]

                self.train_gen = make_generator_w_aug(self.train_img_files,
                                                      self.train_mask_dir,
                                                      self.train_bs,
                                                      self.aug_img_files,
                                                      self.aug_mask_dir
                                                      )

            else:
                self.train_gen = make_generator(self.train_img_files,
                                                self.train_mask_dir,
                                                self.train_bs,
                                                )

        else:
            self.train_imgs = np.load(self.cfg["train"]["input_file"])
            self.train_masks = np.load(self.cfg["train"]["mask_file"])


    def __load_val_data(self):
            #Currently, code does not support val generator. But this is a small change and can be done if needed later
            self.val_img_files = glob.glob(self.cfg["val"]["image_dir"] + "/*.npy")
            self.val_mask_dir = self.cfg["val"]["mask_dir"]
            self.val_imgs = np.load(self.cfg["val"]["input_file"])
            self.val_masks = np.load(self.cfg["val"]["mask_file"])
            self.val_idxs = random.sample(range(self.val_imgs.shape[0]), 200)
            self.val_imgs = self.val_imgs[self.val_idxs]
            self.val_masks = self.val_masks[self.val_idxs]

    def __load_test_data(self):
        assert (self.cfg["prepare_data"]["stride_dim"][0] < self.cfg["prepare_data"]["patch_dim"][0])
        if self.cfg["prepare_data"]["extract_patches"]:
            self.norm_params_df = pd.read_csv(self.cfg["test"]["normalization_params"], sep="\t", index_col=0)
            self.ch_mean = self.norm_params_df.loc["Mean", :].to_numpy()
            self.ch_std = self.norm_params_df.loc["Std", :].to_numpy()
            self.test_df = pd.read_csv(self.cfg["prepare_data"]["test_data"], sep="\t", index_col=0)
        #Only during development.
            # temp_df = self.test_df[:10]
            # pdb.set_trace()
            # self.test_df = temp_df

    def __get_gan_data_batch__(self):
        real_imgs, real_masks = next(self.train_gen)
        if self.disc_arch_name == "patchGAN":
            out_shape = list(self.discriminator_model.output.shape[1:-1])
            out_shape.insert(0, self.train_bs)
        elif self.disc_arch_name == "basic_discriminator":
            out_shape = [self.train_bs, 1]
        else:
            sys.exit("Unsupported Discriminator")
        real_labels = self.real_label * np.ones(shape=out_shape)
        fake_labels = self.fake_label * np.ones(shape=out_shape)

        return real_imgs, real_masks, real_labels, fake_labels


    def __train_one_epoch__(self):
        hist_df = pd.DataFrame()
        d_loss_real_epoch = []
        d_loss_fake_epoch = []
        gen_adv_loss_epoch = []
        gen_mae_loss_epoch = []
        for batch_idx in range(self.steps_per_epoch):
            real_imgs, real_masks, real_labels, fake_labels = self.__get_gan_data_batch__()
            # First Train Discriminator on Real and Fake Masks
            d_real_hist = self.discriminator_model.train_on_batch(real_masks, real_labels, return_dict=True)
            d_loss_real = d_real_hist["loss"]
            d_acc_real = d_real_hist["accuracy"]

            fake_gen_masks = self.generator_model(real_imgs)
            d_fake_hist = self.discriminator_model.train_on_batch(fake_gen_masks, fake_labels,  return_dict=True)
            d_loss_fake = d_fake_hist["loss"]
            d_acc_fake = d_fake_hist["accuracy"]

            # Then Train Generator via Composite Model on Fake Masks but with adversarial labels
            gen_hist = self.composite_gan_model.train_on_batch([real_imgs], [real_labels, real_masks], return_dict=True)
            gen_adv_loss = gen_hist["model_loss"]
            gen_mae_loss = gen_hist["model_1_loss"]
            print(f"batch_idx: {batch_idx}, d_real: {d_loss_real}, d_fake: {d_loss_fake}, adv_loss: {gen_adv_loss}, mae:{gen_mae_loss}")

            #Record Various Losses
            d_loss_real_epoch.append(d_loss_real)
            d_loss_fake_epoch.append(d_loss_fake)
            gen_adv_loss_epoch.append(gen_adv_loss)
            gen_mae_loss_epoch.append(gen_mae_loss)
            hist_df.loc[batch_idx, "d_loss_real"] = d_loss_real
            hist_df.loc[batch_idx, "d_loss_fake"] = d_loss_fake
            hist_df.loc[batch_idx, "gen_adv_loss"] = gen_adv_loss
            hist_df.loc[batch_idx, "gen_mae_loss"] = gen_mae_loss
            hist_df.loc[batch_idx, "d_acc_real"] = d_acc_real
            hist_df.loc[batch_idx, "d_acc_fake"] = d_acc_fake

        d_loss_real_mean = np.asarray(d_loss_real_epoch).mean()
        d_loss_fake_mean = np.asarray(d_loss_fake_epoch).mean()
        gen_adv_loss_mean = np.asarray(gen_adv_loss_epoch).mean()
        gen_mae_loss_mean = np.asarray(gen_mae_loss_epoch).mean()
        return (d_loss_real_mean, d_loss_fake_mean, gen_adv_loss_mean, gen_mae_loss_mean, hist_df)


    def save_models(self, ep_idx):
        # save the generator model
        filename1 = f'ckpt_generator_{self.gen_arch_name}_{ep_idx}.h5'
        self.generator_model.save_path = os.path.join(self.mdl_dir, filename1)
        self.generator_model.save(self.generator_model.save_path)
        print('>Saved: %s ' % filename1)


    def __do_validation__(self):
        val_gen_masks = self.generator_model(self.val_imgs)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        mae = tf.keras.losses.MeanAbsoluteError()
        # compute val loss. Maybe do mae instead of bce
        val_bce_loss = bce(self.val_masks, val_gen_masks).numpy()
        val_mae_loss = mae(self.val_masks, val_gen_masks).numpy()
        # val

        res_dict, _ = compute_perf_metrics(self.val_masks, val_gen_masks,
                                           labels=self.cfg["data"]["labels"],
                                           target_names=self.cfg["data"]["target_names"],
                                           threshold_confusion=self.threshold_confusion
                                           )
        val_dice = res_dict['Dice']
        return (val_bce_loss, val_mae_loss, val_dice)


    def train_model(self):
        self.full_hist_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        for ep_idx in range(self.num_epochs):
            d_loss_real_mean, d_loss_fake_mean, gen_adv_loss_mean, gen_mae_loss_mean, hist_df = self.__train_one_epoch__()
            val_bce_loss, val_mae_loss, val_dice = self.__do_validation__()

            self.full_hist_df = pd.concat([self.full_hist_df, hist_df], axis='index')
            self.full_hist_df = self.full_hist_df.reset_index(drop=True)
            print(f"epoch={ep_idx}/{self.num_epochs}, d_real: {d_loss_real_mean}, d_fake: {d_loss_fake_mean}")
            print(f"epoch={ep_idx}/{self.num_epochs}, adv_loss: {gen_adv_loss_mean}, mae_loss: {gen_mae_loss_mean}")
            print("\n\n")

            # Save the model after every epoch
            self.save_models(ep_idx)
            self.val_df.loc[ep_idx, "val_bce_loss"] = val_bce_loss
            self.val_df.loc[ep_idx, "val_mae_loss"] = val_mae_loss
            self.val_df.loc[ep_idx, "val_dice"] = val_dice
            self.val_df.loc[ep_idx, "Generator_Path"] = self.generator_model.save_path

        #save history at end of training
        self.full_hist_df.to_csv(os.path.join(self.train_dir, "train.tsv"), sep="\t")
        #save the best model with highest dice coeff
        best_row = self.val_df.iloc[self.val_df["val_dice"].argmax()]
        shutil.copy2(src=best_row["Generator_Path"], dst=self.final_mdl_dir)
        self.best_gen_path = best_row["Generator_Path"]


# load the generator
# predict the patches using the model
    def model_predict_by_patches(self):
        #only for mobilenet
        if 'mobilenet' in self.gen_arch_name:
            nn_model = tf.keras.models.load_model(self.best_gen_path, custom_objects={'relu6': relu6})
        else:
            nn_model = tf.keras.models.load_model(self.best_gen_path)

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
                if img_name == "00252_p2.tif":
                    # image: 00252_p2.tif shape=(1040, 1388,4).
                    # this is weird and unexpected, hence, changing it to grayscale
                    test_img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

                else:
                    test_img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
                test_mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
                print(f"{img_name}, img: {test_img.shape}, mask: {test_mask.shape}")
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

                test_pred_patches = nn_model.predict(test_img_patches)
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
                                            labels=self.cfg["data"]["labels"],
                                            target_names=self.cfg["data"]["target_names"],
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
                                            labels=self.cfg["data"]["labels"],
                                            target_names=self.cfg["data"]["target_names"],
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





















