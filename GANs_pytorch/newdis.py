import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import init
from torchvision import transforms
import torchvision.utils as vutils
import time
import os
from numpy import array
from sklearn.metrics import precision_score
from dataLoad import CustomDataset
from torch.autograd import Variable
from PIL import *
import cv2

'''
TO ADD:
Also check the evaluation criteria from the gans paper
'''

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp / 2 + 0.5  # unnormalize
    npimg = inp.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(1)  # pause a bit so that plots are updated

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def mobiConv1(in_channels, out_channels, stride=1,
              padding=1, bias=True, groups=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=in_channels)


def mobiConv2(in_channels, out_channels, stride=1,
              padding=0, bias=True, groups=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):

    if mode == 'transpose':

        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels

        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):

    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):

        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        # conv1 to increase the number of channels

        # Resnet 2 3*3 operations

        # conv2 to retain the dimenstions
        
        ## check for LeakyRelu and Batch Normalization [removed] and max pool
        ## GANS paper use LEAKY RELU and BATCH NORM.
        ## UNET use RELU and NO BATCH NORM.
        ## current code uses LEAKY RELU and BATCH NORM.

        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.mobinet1 = mobiConv1(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.mobinet2 = mobiConv2(self.out_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.mobinet3 = mobiConv1(self.out_channels, self.out_channels)
        self.bn4 = nn.BatchNorm2d(self.out_channels)
        self.mobinet4 = mobiConv2(self.out_channels, self.out_channels)
        self.bn5 = nn.BatchNorm2d(self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn6 = nn.BatchNorm2d(self.out_channels)

        #self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2,stride=2, padding=0) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # if self.pooling:

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual1 = x

        #print("Before 1 ")
        #print(x.size())
        x = self.mobinet1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.mobinet2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mobinet3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.mobinet4(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        #print("Before 2")
        #print(x.size())
        x += residual1
        #print("Before 3")
        #print(x.size())
        x = self.conv2(x)
        x = self.bn6(x)
        x = self.relu(x)
        #print("Before 4")
        #print(x.size())
        
        
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):

    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.relu = nn.LeakyReLU(0.2)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.mobinet1 = mobiConv1(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.mobinet2 = mobiConv2(self.out_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.mobinet3 = mobiConv1(self.out_channels, self.out_channels)
        self.bn4 = nn.BatchNorm2d(self.out_channels)
        self.mobinet4 = mobiConv2(self.out_channels, self.out_channels)
        self.bn5 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn6 = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        #print("UPBefore ")
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual1 = x
		#print("UPBefore 1 ")
        #print(x.size())
        x = self.mobinet1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.mobinet2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mobinet3(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.mobinet4(x)
        x = self.bn5(x)
        x = self.relu(x)
        #print("UPBefore 2 ")
        #print(x.size())
        x += residual1

        #print("UPBefore 3 ")
        #print(x.size())
        x = self.conv2(x)
        x = self.bn6(x)
        x = self.relu(x)

        #print("UPBefore 4 ")
        #print(x.size())
        return x

#According to pix-pix
## ACCORDING TO GANS PAPER
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)    
        self.relu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 1)
        #self.bn4 = nn.BatchNorm2d(self.out_channels)
        #self.relu5 = nn.LeakyReLU(0.2)
        self.fcClass = nn.Linear(225, 7)
        self.fcDis = nn.Linear(225, 1)
        
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        #x = self.bn4(x)
        #x = self.relu5(x)
        
        flat=x.view(-1,225)
        fcClass=self.fcClass(flat)  
        fcDis=self.fcDis(flat)
        
        classes=self.logSoftmax(fcClass)
        predic=self.softmax(fcClass)
        #check
        realfake=self.sigmoid(fcDis).view(-1, 1).squeeze(1)
       
        #realfake=self.sigmoid(fcDis)
        
        return realfake, classes, predic
    
    #weight initial problem
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.normal_(0.0, 0.02)

        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        
#Applying a Sigmoid to the end
#using BCE loss, not L1
class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from thene encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def __init__(self, num_classes, in_channels=1, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

            # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            #print("Conv "+str(i))
            #print("Input "+str(ins) +" Output "+str(outs))
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()


    '''
    def init_weights(m):
        
        if type(m) == nn.Conv:
            torch.nn.init.normal_(0.0, 0.02)

        elif type(m) == nn.BatchNorm:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
   '''
    @staticmethod
    def weight_init(m):
        #from pix to pix
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    

    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        x = self.sigmoid(x)
        return x
    
def plotting1():
    x,y=[],[]
    f = open('/home/vit1/Downloads/hep2/newLossFun/model/weights.txt', "r").readlines()
    for a in f:
        p = a.split(',')
        x.append(int(p[1]))
        y.append(float(p[-1].strip()))

    x, y = array(x), array(y)
    plt.plot(x, y)
    plt.savefig('/home/vit1/Downloads/hep2/newLossFun/model/precision_score.png')
    
def plotting2():
    x,y=[],[]
    f = open('/home/vit1/Downloads/hep2/newLossFun/model/loss.txt', "r").readlines()
    for a in f:
        p = a.split(',')
        x.append(int(p[1]))
        y.append(float(p[-1].strip()))
    x, y = array(x), array(y)
    plt.plot(x, y)
    plt.savefig('/home/vit1/Downloads/hep2/newLossFun/model/loss_graph.png')

if __name__ == "__main__":
    # cd to file location
    # python code.py --d data --r results --s saved --lr .0002 --b 0.5
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")

    ap.add_argument("-r", "--results", required=True, help="path to results")

    ap.add_argument("-s", "--saved", required=True, help="path to all saved")

    ap.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

    ap.add_argument('--b', type=float, default=0.5, help='beta1 for adam. default=0.5')

    args = vars(ap.parse_args())

    print(args["dataset"])
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

    csv_path='/home/vit1/Downloads/hep2/train.csv'
    img_path_ground='/home/vit1/Downloads/hep2/ground'
    customDatasetGround=CustomDataset(csv_path, img_path_ground, transform)
    
    print("NUM OF IMAGES")
    print(len(customDatasetGround))
   
    groundloader = torch.utils.data.DataLoader(customDatasetGround, batch_size=1,
                                              shuffle=False, num_workers=1)
    
    
    img_path_spec='/home/vit1/Downloads/hep2/spec'
    customDatasetSpec=CustomDataset(csv_path, img_path_spec, transform)
   
    specloader = torch.utils.data.DataLoader(customDatasetSpec, batch_size=1,
                                              shuffle=False, num_workers=1)
    
    
    '''
    # TO SHOW THE IMAGES INITIALLY
    #print(len(image_datasets[0][0]))
    dataiter = iter(trainloader)
    images,labels = dataiter.next()
     show images
    out = torchvision.utils.make_grid(images)
    imshow(out)
    model = UNet(3, depth=3, merge_mode='add')
    x = Variable(images)
    outn = model(x)
    '''

    netGen = UNet(num_classes=1, in_channels=1, depth=3, merge_mode='add')
    netGen.to(device)
    
    #netGen.init_weights()
    netDis = Discriminator()
    netDis.to(device)
    netDis.init_weights()
    
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()
    genL1_criterion = nn.L1Loss()
    
    gen_criterion = nn.BCELoss() #when using SIGMOID IN the last layer 
    
    batchSize=1
    numClass=7
    dis_label = torch.FloatTensor(batchSize)
    aux_label = torch.LongTensor(batchSize)
    aux_label_GEN = torch.LongTensor(batchSize)
    real_label = 1
    fake_label = 0
    
    dis_label = Variable(dis_label).to(device)
    aux_label = Variable(aux_label).to(device)
    aux_label_GEN = Variable(aux_label_GEN).to(device)
    
    optimizerD = optim.Adam(netDis.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netGen.parameters(), lr=0.0002, betas=(0.5, 0.999))


    timeElapsed = []
    epochs=60

    save_folder_name='/home/vit1/Downloads/hep2/newLossFun/SAVE'

    for epoch in range(epochs):  # loop over the dataset multiple time
        ps = 0
        print("# Starting epoch [%d/%d]..." % (epoch, epochs))
        running_loss = 0.0
        
        grounditer = iter(groundloader)
        try:
            for i, data in enumerate(specloader, 0):
                # get the inputs
                specImg, labelSpec = data
                specImg = specImg.to(device)
                labelSpec = labelSpec.to(device)
                #print(specImg.size())
                #groundImg is just actual segmentation mask.
                #label is the class label (pattern) of the image
                groundImg, label = grounditer.next()
                groundImg = groundImg.to(device)
                label = label.to(device)
                # zero the parameter gradients

                netDis.zero_grad()
                #out = torchvision.utils.make_grid(groundImg)
                #imshow(out)

                #out1= torchvision.utils.make_grid(specImg)
                #imshow(out1)

                #LABELS
                dis_label.data.resize_(batchSize).fill_(real_label)
                aux_label.data.resize_([1,7]).copy_(label)

                #---------------------------------------------------
                #train discriminator for groundTruth
                dis_output, aux_output, predic=netDis(groundImg)
                aux_label=torch.max(aux_label, 1)[1]
                print("----AUX OUTPUT NETWORK----")
                print(predic)
                print("AUX_LABEL ACTUAL")
                print(aux_label)
                #print("AUX_LABEL OUTPUT")
                #print(aux_output)
                #print("--------")
                #print("Original OR Fake ACTUAL")
                #print(dis_label)
                #print("Original OR Fake From SYSTEM")
                #print(dis_output)
                dis_errD_real = dis_criterion(dis_output, dis_label)
                aux_errD_real = aux_criterion(aux_output, aux_label)
                errD_real = dis_errD_real + aux_errD_real 
                errD_real.backward()
                #dis_errD_real.backward(retain_graph=True)
                #aux_errD_real.backward()
                #optimizerD.step()
                #netDis.zero_grad()
                
                #--------------------------------------------------------- 
                
               

                #LABELS
                #check random for aux_label
                label = np.random.randint(0, numClass, batchSize)
                aux_label_GEN.data.resize_(batchSize).copy_(torch.from_numpy(label))
                #print("---RANDUM AUX----")
                #print(aux_label_GEN)
                #print("------")
                dis_label.data.fill_(fake_label)  
                
                #--------------------------------------------------------- 
                #train discriminator for output from generator
                #netGen.zero_grad()
                outputGen = netGen(specImg)
                outputGen1 = outputGen.to(torch.device("cpu"))

                #SAVING IMAGES START
                img=outputGen1.data.numpy()
                #img.flags.writeable=True
                img[img >= 0.5] = 1
                img[img < 0.5] = 0
                img_cont = img * 255
                #img_cont_np = np.asarray(img_cont)
                #img_cont_np = img_cont.astype('uint8')
                #print(img_cont.shape)
                img_cont.resize((256, 256))
                #print(img_cont.shape)
                #print(img_cont)
                img_cont = Image.fromarray(img_cont)
                img_cont = img_cont.convert("L")
                desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
                # Create the path if it does not exist
                if not os.path.exists(desired_path):
                    os.makedirs(desired_path)
                # Save Image!
                export_name = str(i) + '.png'
                img_cont.save(desired_path + export_name)

                #accuracy_check(original_msk, pred_msk)
                #SAVING IMAGES ENDS
                print("YOLO")
                dis_output, aux_output, predic = netDis(outputGen)
                dis_errD_fake = dis_criterion(dis_output, dis_label)
                aux_errD_fake = aux_criterion(aux_output, aux_label_GEN)
                
                errD_fake = dis_errD_fake + aux_errD_fake
                #err = errD_real + errD_fake
                #dis_errD_fake.backward(retain_graph=True)
                #print("YOLO")
                #aux_errD_fake.backward(retain_graph=True)
                errD_fake.backward()
                
                optimizerD.step()
                #netDis.zero_grad()
                
                #--------------------------------------------------------- 


                #LABEL
                dis_label.data.fill_(real_label)

                
                #--------------------------------------------------------- 
                print("YOLO1")
                #train generator
                netGen.zero_grad()
                outputGen = netGen(specImg)
                dis_label.data.resize_(batchSize).fill_(real_label)
                
                dis_output, aux_output, predic = netDis(outputGen)
                
                dis_errG = dis_criterion(dis_output, dis_label)
                aux_errG = aux_criterion(aux_output, aux_label)

                gen_L1loss=genL1_criterion(outputGen, groundImg)
                #gen_loss=gen_criterion(outputGen, groundImg)#when using SIGMOID IN THE END
                
                L1_lambda=100
                errG = dis_errG + aux_errG + (L1_lambda * gen_L1loss )
                #errG = dis_errG + aux_errG + gen_loss #When using Sigmoid in the end
                #errG = dis_errG + aux_errG + gen_loss
                
                errG.backward()
                print("YOLO2")
                optimizerG.step()
                #netGen.zero_grad()

                #--------------------------------------------------------- 
                


                #outputs.view(65536)
                #vutils.save_image(torch.FloatTensor(outputs), os.getcwd()+ "/results/test_output.png")
                # print statistics
                running_loss += errG.item()
                
                print(str(running_loss))
                print("LOSS"+ str(errG.item()))
                f = open("/home/vit1/Downloads/hep2/newLossFun/model/weights.txt","a+")
 
                '''
                outputGen2 = outputGen.to(torch.device("cpu"))
                outputGen2 = outputGen2.view(65536)
                outputGen2 = outputGen2.data.numpy()
                outputGen2 = np.round(outputGen2)
                             
                groundImg1 = groundImg.to(torch.device("cpu"))   
                groundImg1 = groundImg1.view(65536)
                groundImg1 = groundImg1.data.numpy()*-1
                
                '''
    
                img1 = cv2.imread("/home/vit1/Downloads/hep2/ground/"+str(i)+".png",0)
                
                img2 = cv2.imread(desired_path + export_name,0)

                ret,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
                
                ret,thresh2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
                
                
                ps += precision_score(thresh1, thresh2, average='macro')
                print("Precision- ")
                print(precision_score(thresh1, thresh2, average='macro'))
                print("\n\n\n")
                #f.write("loss: "+str(epoch)+str(ps)+"\n")
                
        except Exception as e: print(e)
        #except:
         #   pass
        ps/=20160
        f = open('/home/vit1/Downloads/hep2/newLossFun/model/weights.txt','a+')
        f.write("ps"+","+str(epoch+1)+","+str(ps)+"\n")
        f.flush()
        f.close()
        f = open('/home/vit1/Downloads/hep2/newLossFun/model/loss.txt','a+')
        f.write("Loss"+","+str(epoch+1)+","+str(running_loss)+"\n")
        f.flush()
        f.close()

    print('Finished Training')
    
    plotting1()
    plotting2()

    torch.save(netGen.state_dict(), '/home/vit1/Downloads/hep2/newLossFun/model/model_gen.pt')  # save model

    print('Model for generator saved')

    torch.save(netDis.state_dict(), '/home/vit1/Downloads/hep2/newLossFun/model/model_dis.pt')  # save model

    print('Model for Discriminator saved')

    print('Saved graph')


# In[ ]:


a = 42 /20
print(a)


# In[ ]:





# In[ ]:




