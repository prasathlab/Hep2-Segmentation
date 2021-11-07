from GANs_pytorch.data_utils import *
from arg_parser import *
from file_utils import *
from GANs_pytorch.gan_unet import *
from GANs_pytorch.gan_discriminator import *
from torchsummary import summary
from tqdm import tqdm
import sys
import yaml
import pdb

class HEP2_GAN_app:
    def __init__(self, yaml_filepath, args, sys_argv=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        with open(yaml_filepath, 'r') as stream:
            try:
                self.cfg = yaml.load(stream)
                print(self.cfg)
            except yaml.YAMLError as exc:
                print(exc)

        self.__get_hyperparams__()
        self.__get_arch_prefix__()

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
            self.train_img_dir = self.cfg["train"]["image_dir"]
            self.train_mask_dir = self.cfg["train"]["mask_dir"]
            self.train_dataset = Hep2TrainValDataset(self.train_img_dir, self.train_mask_dir)
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.train_bs,
                                               shuffle=True,
                                               num_workers=0
                                               )
        if "val" in self.cfg:
            self.val_img_dir = self.cfg["val"]["image_dir"]
            self.val_mask_dir = self.cfg["val"]["mask_dir"]
            self.val_dataset = Hep2TrainValDataset(self.val_img_dir, self.val_mask_dir)
            self.train_dataloader = DataLoader(self.val_dataset,
                                               batch_size=self.train_bs,
                                               shuffle=True,
                                               num_workers=0
                                               )
        if "test" in self.cfg:
            pass
            # TODO: Development of Test Set and Procedure.
            # self.__load_test_data()
            # self.threshold_confusion = float(self.cfg["test"]["threshold_confusion"])

        pdb.set_trace()
        # Build the generator model
        self.gen_model = UNet(num_classes=1,
                              in_channels=1,
                              depth=3,
                              merge_mode='add')
        self.gen_model = self.gen_model.float()
        summary(self.gen_model, input_size=(self.n_channels, self.height, self.width))
        pdb.set_trace()
        # Build the Discriminatior model
        self.discrimiator_model = Discriminator()
        self.discrimiator_model = self.discrimiator_model.float()
        summary(self.discrimiator_model, input_size=(self.n_channels, self.height, self.width))

        pdb.set_trace()
        #Setup the train loop. Save model, make train/val curves.
        self.discrimiator_model.to(self.device)
        self.discrimiator_model.init_weights()
        pdb.set_trace()
        # Loss for Real or Fake
        self.dis_criterion = nn.BCELoss()
        #Loss for the class label of the image
        self.aux_criterion = nn.NLLLoss()
        #L1 criterion to get generated pixels right
        self.genL1_criterion = nn.L1Loss()
        pdb.set_trace()
        #what is this loss for..Need to check
        self.gen_criterion = nn.BCELoss()  # when using SIGMOID IN the last layer
        self.real_label = 1
        self.fake_label = 0
        pdb.set_trace()
        # numClass = 7
        # self.dis_label = torch.FloatTensor(batchSize)
        # aux_label = torch.LongTensor(batchSize)
        # aux_label_GEN = torch.LongTensor(batchSize)
        #
        #
        # dis_label = Variable(dis_label).to(device)
        # aux_label = Variable(aux_label).to(device)
        # aux_label_GEN = Variable(aux_label_GEN).to(device)

        self.optimizerD = optim.Adam(self.discrimiator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        #Setup the predict by patches method

        pdb.set_trace()
        debug = 1

    def __get_arch_prefix__(self):
        self.prefix = self.arch_name
        if self.pretrained_w:
            self.prefix = self.prefix + "_" + "PT"
            if self.fine_tune:
                self.prefix = self.prefix + "_" + "FT"
            else:
                self.prefix = self.prefix + "_" + "NFT"
        else:
            # This is just to make sure there is never a case of NPT and NFT
            assert (not self.pretrained_w and self.fine_tune)
            self.prefix = self.prefix + "_" + "NPT" + "_" + "FT"

    def __get_hyperparams__(self):
        # HyperParams
        self.train_bs = self.cfg["train"]["batch_size"]
        self.steps_per_epoch = self.cfg["train"]["steps_per_epoch"]
        self.num_epochs = self.cfg["train"]["num_epochs"]
        self.lr = float(self.cfg["train"]["learning_rate"])
        self.loss = self.cfg["train"]["loss"]
        self.optimizer_name = self.cfg["train"]["optimizer"]
        self.arch_name = self.cfg["project"]["arch"]
        self.height = self.cfg["data"]["height"]
        self.width = self.cfg["data"]["width"]
        self.n_channels = self.cfg["data"]["n_channels"]
        # Currently only below shape is supported.
        self.input_shape = (self.height, self.width, self.n_channels)
        self.pretrained_w = self.cfg["train"]["pretrained_w"]
        self.fine_tune = self.cfg["train"]["fine_tune"]
        self.n_classes = len(self.cfg["data"]["labels"])
        self.L1_weight = 100


    def __train_discriminator__(self, real_masks, fake_gen_masks ):
        pdb.set_trace()
        real_labels = self.real_label * np.ones(self.train_bs)
        real_labels = torch.from_numpy(real_labels).float().to(self.device)
        self.discrimiator_model.zero_grad()
        pred_real_labels = self.discrimiator_model(real_masks.float())
        real_err_disc = self.dis_criterion(pred_real_labels, real_labels)
        real_err_disc.backward()

        pdb.set_trace()
        pred_fake_labels = self.discrimiator_model(fake_gen_masks.float())
        fake_gen_labels = self.fake_label * np.ones(self.train_bs)
        fake_gen_labels = torch.from_numpy(fake_gen_labels).float().to(self.device)
        fake_err_disc = self.dis_criterion(pred_fake_labels, fake_gen_labels)
        fake_err_disc.backward()

        pdb.set_trace()
        full_disc_loss = real_err_disc + fake_err_disc
        self.optimizerD.step()

        pdb.set_trace()
        return full_disc_loss


    def train_model(self):
        save_folder_name = '/home/vit1/Downloads/hep2/newLossFun/SAVE'
        pdb.set_trace()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple time
            ps = 0
            print("# Starting epoch [%d/%d]..." % (epoch, self.num_epochs))
            running_loss = 0.0

            # Load img, mask from train_dataloader
            # Give img as input to generator and get mask as output
            # For img, mask from dataset--> label as real and train discriminator
            # For img, mask from generator --> label as fake and train discriminator
            # For training generator:
            pdb.set_trace()
            batch_iter = tqdm(self.train_dataloader, total= self.steps_per_epoch)
            print(f"\n\n epoch: {epoch} of {self.num_epochs} \n\n ")
            for batch_idx, batch_data in enumerate(self.train_dataloader, 0):
                pdb.set_trace()
                real_imgs, real_masks = batch_data
                pdb.set_trace()
                fake_gen_masks = self.gen_model(real_imgs.float())
                pdb.set_trace()

                #train the discriminator. For now ignore the class label (pattern) of the image.
                #train discriminator for groundTruth masks

                loss_disc_total = self.__train_discriminator__(real_masks, fake_gen_masks)
                pdb.set_trace()
                #Train Discriminator on Generated (Fake) Masks
                #loss_disc_fake = self.__train_discriminator__(fake_gen_masks, fake_gen_labels)


                #Train generator
                #First take the generated masks, give it to discriminator but with adversarial label i.e.
                #   intentionally label them as real. Compute adversarial loss
                pdb.set_trace()
                self.gen_model.zero_grad()
                disc_outputs = self.discrimiator_model(fake_gen_masks)
                pdb.set_trace()
                adversarial_labels = self.real_label*np.ones(self.train_bs)
                pdb.set_trace()
                adversarial_labels = torch.from_numpy(adversarial_labels).float().to(self.device)
                adversarial_loss = self.dis_criterion(disc_outputs, adversarial_labels)
                pdb.set_trace()
                gen_L1loss = self.genL1_criterion(fake_gen_masks, real_masks)
                pdb.set_trace()
                errG = adversarial_loss + (self.L1_weight * gen_L1loss)
                pdb.set_trace()
                errG.backward()
                pdb.set_trace()
                self.optimizerG.step()
                pdb.set_trace()
                print(f"batch_idx: {batch_idx}, loss_disc_real={loss_disc_real}, loss_disc_fake={loss_disc_fake} ")
                print(f"batch_idx: {batch_idx}, adversarial_loss={adversarial_loss}, gen_L1loss={gen_L1loss} ")
                pdb.set_trace()








# pdb.set_trace()
# yaml_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation/GANs_pytorch/temp_exp.yaml"
# gan_app = HEP2_GAN_app(yaml_path)
# pdb.set_trace()
# img, mask = gan_app.train_dataset[100]
# pdb.set_trace()
# debug = 1


