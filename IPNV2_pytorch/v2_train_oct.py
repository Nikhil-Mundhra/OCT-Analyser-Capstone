import os
#setting GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import logging
import sys

import model
import utils
import shutil
import natsort
from options.train_options import TrainOptions
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
from torchsummary import summary
from skimage.transform import resize
from scipy.special import softmax
import imageio.v2 as imageio
from PIL import Image

from sklearn import metrics
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
#from reliability_diagrams import *
from plot_reliability_diagram import *


def compute_brier(y_true, y_prob):
    """
    Compute the Brier Score.

    Parameters:
        - y_true: NumPy array, true binary labels (0 or 1).
        - y_prob: NumPy array, predicted probabilities.

    Returns:
        - float: Brier Score.
    """
    return np.mean((y_true - y_prob) ** 2)


def compute_nll(y_true, y_log_prob):
    """
    Compute the Negative Log-Likelihood.

    Parameters:
        - y_true: NumPy array, true binary labels (0 or 1).
        - y_log_prob: NumPy array, predicted log probabilities.

    Returns:
        - float: Negative Log-Likelihood.
    """
    try:
        return -np.mean(y_log_prob[np.arange(len(y_true)), y_true])
    except:
        return float('inf')
    
    

def plot_dice(val_dice_scores):
    filename = "_val_dice_scores.png"

    arr = []
    for i in range(len(val_dice_scores)):
        arr.append(val_dice_scores[i])
    plt.plot(np.array(arr), 'r', label='Val')
  

    plt.title('Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend(loc='best')
    
    filename = os.environ["SLURM_JOB_ID"] + filename
    plt.savefig(os.path.join(opt.saveroot, 'plots', filename))
    plt.close()
    
    
def plot_loss(train_loss_value):
    filename = "_train_loss_curves.png"
    
    arr = []
    for i in range(len(train_loss_value)):
        arr.append(train_loss_value[i])

    plt.plot(np.array(arr), 'b', label='Train')

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    filename = os.environ["SLURM_JOB_ID"] + filename
    plt.savefig(os.path.join(opt.saveroot, 'plots', filename))
    plt.close()


################################
#          DSC++ loss          #
################################

def dice_plus_loss(gamma=2):
    """Dice++ loss function as described in the paper.

    Args:
        gamma (float): controls the degree of penalization for FN and FP predictions.
                       Higher gamma values favor low-confidence predictions.
    """
    def loss_function(y_true, y_pred):
        epsilon = 1e-7
        axis = tuple(range(1, len(y_true.shape)))  # Identify axis for sum reduction

        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_true = torch.clamp(y_true, epsilon, 1 - epsilon)

        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum((y_true * (1 - y_pred))**gamma, dim=axis)
        fp = torch.sum(((1 - y_true) * y_pred)**gamma, dim=axis)
        dice_class = (2 * tp + epsilon) / (2 * tp + fn + fp + epsilon)
        loss = torch.mean(1 - dice_class)

        return loss

    return loss_function


def train_net(net,device):
    logging.info(net)
    #train setting
    interval=opt.save_interval
    train_num = opt.train_ids[1] - opt.train_ids[0]
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    BLOCK_SIZE = opt.block_size
    val_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]))
    cube_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))
    best_valid_dice=0
    # model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    # best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    model_save_path = os.path.join(opt.saveroot, 'checkpoints_oct')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model_oct')
    # Read Data
    print("Start Setup dataset reader")
    train_records, validation_records = readData.read_dataset(opt.dataroot, opt.train_ids, opt.val_ids, opt.modality_filename)
    print("Setting up dataset reader")
    train_dataset_reader = BatchDataReader.BatchDatset(train_records, opt.modality_filename, opt.data_size, opt.block_size, opt.in_channels,
                                                       opt.batch_size, train_num, "train", opt.saveroot)
    
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    
    # Plot
    train_loss_values = []
    val_loss_values = []
    train_dice_values = []
    val_dice_values = []
    
    #Setting Loss
    criterion = nn.CrossEntropyLoss() # Dice combined with NLL loss?
    #Start train
    for itr in range(0, opt.max_iteration):
        net.train()
        train_images, train_annotations = train_dataset_reader.read_batch_random_train()
        
        train_annotations[train_annotations==100]=0
        train_annotations[train_annotations>=1]=1
        #train_images, train_annotations = train_dataset_reader.read_batch_normal_train()

        #if itr % 2 ==0:
        #    train_images,train_annotations = train_dataset_reader.read_batch_normal_train()
        #else:
        #    train_images, train_annotations= train_dataset_reader.read_batch_random_train()

        train_images = train_images.to(device=device, dtype=torch.float32)/255.0
        train_annotations = train_annotations.to(device=device, dtype=torch.long)

        
        optimizer.zero_grad()
        pred, _= net(train_images)
        loss = criterion(pred, train_annotations)
        
        # loss = dice_plus_loss(1.)(train_annotations, pred)
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            logging.info(str(itr) + str(" ") + str(loss.item()))
            print(itr, loss.item())
            train_loss_values.append(loss.item())
            
        #Start Val
        with torch.no_grad():
            if itr % interval==0: # every 500 iteration
                #Save model
                torch.save(net.state_dict(),
                           os.path.join(model_save_path,f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
               
                #Calculate validation Dice
                val_Dice_sum = 0
                net.eval()
                valids = opt.val_ids
                cubelist0 = os.listdir(os.path.join(opt.dataroot, 'val', opt.modality_filename[0]))
                cubelist0 = natsort.natsorted(cubelist0)
                # cubelist = cubelist0[valids[0]:valids[1]]
                
                print("len of cubelist: ", len(cubelist0))
                for kk, cube in enumerate(cubelist0):
                    bscanlist = os.listdir(os.path.join(opt.dataroot, 'val', opt.modality_filename[0], cube))
                    bscanlist = natsort.natsorted(bscanlist)
                    for i, bscan in enumerate(bscanlist):
                        for j, modal in enumerate(opt.modality_filename):
                            if modal != opt.modality_filename[-1]: # label
                                #image_arr = np.array(Image.open(os.path.join(opt.dataroot, 'val', modal, cube, bscan)))
                                # cube_images[0, j, :, :, i] = np.array(Image.fromarray(image_arr).resize((DATA_SIZE[1], BLOCK_SIZE[0]), Image.NEAREST))
                                cube_images[0, j, :, :, i] = np.array(resize(imageio.imread(os.path.join(opt.dataroot, 'val', modal, cube, bscan))/255.0, (BLOCK_SIZE[0], DATA_SIZE[1]), order=1))
                    
                    result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
                    #label= Image.open(os.path.join(opt.dataroot, 'val', opt.modality_filename[opt.in_channels], f'{cube}.bmp'))
                    label_arr = imageio.imread(os.path.join(opt.dataroot, 'val', opt.modality_filename[opt.in_channels], f'{cube}.bmp'))
                    label_arr = np.rot90(label_arr, 3) # rotate three times = rotate clockwise once
                    label_arr[label_arr==100]=0
                    label_arr[label_arr>=1]=1
                    
                    for i in range(0, DATA_SIZE[1],BLOCK_SIZE[1]):
                        for j in range(0, DATA_SIZE[2],BLOCK_SIZE[2]):
                            val_images[0, :, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = cube_images[0, :, :,i:i + BLOCK_SIZE[1],j:j + BLOCK_SIZE[2]]
                            # logging.info(val_images) value exists
                            
                            images = torch.from_numpy(val_images)
                            # logging.info(images.size()) # [1, 2, 160, 100, 100] ?the meaning of the shape
                            
                            images = images.to(device=device, dtype=torch.float32)
                            
                            pred, _ = net(images)
                            pred_prob = pred.cpu().detach().numpy()
                            pred_prob = softmax(pred_prob, axis=1) # is softmax necessary?
                            
                            # logging.info(pred_prob)
                            
                            pred_argmax = torch.argmax(pred, dim=1) 
                            confidence_prob = np.max(pred_prob, axis=1)
                            confidence = torch.max(pred, dim=1)
                            # logging.info("confidence: " + str(confidence))
                            
                            val_brier = compute_brier(result[i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]], pred_prob[0, 0, :, :])
                            # logging.info("brier: " + str(val_brier))
                            
                            val_nll = compute_nll(result[i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]], np.log(pred_prob[0, 0, :, :]))
                            # logging.info("nll: " + str(val_nll))
                            
                            result[i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]] = result[i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]] + pred_argmax[0, 0, :, :].cpu().detach().numpy()
            
                    # calculate valid loss (individual)?
                    # val_loss = criterion(result, label_arr)
                    val_Dice_sum+= utils.cal_Dice(result, label_arr)
                    


                val_Dice=val_Dice_sum/val_num
                logging.info("Step:{}, Valid_Dice:{}".format(itr, val_Dice))
                print("Step:{}, Valid_Dice:{}".format(itr, val_Dice))
                val_dice_values.append(val_Dice)
                
                #save best model
                if val_Dice > best_valid_dice:
                    temp = '{:.5f}'.format(val_Dice)
                    if not os.path.exists(os.path.join(best_model_save_path, temp)):
                        os.mkdir(os.path.join(best_model_save_path, temp))
                    temp2= f'{itr}.pth'
                    shutil.copy(os.path.join(model_save_path, temp2), os.path.join(best_model_save_path, temp, temp2))

                    model_names = natsort.natsorted(os.listdir(best_model_save_path))
                    #print(len(model_names))
                    if len(model_names) == 4:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                    best_valid_dice = val_Dice
    
   
    # Plot metrics and diagrams
    plot_dice(val_dice_values)
    plot_loss(train_loss_values)
    filename = os.environ["SLURM_JOB_ID"] + "_reliability_diagram"
    # plot_reliabiblity_diagram(y_true=y_true, y_pred=y_pred, n_bins=10, rel_diag_folder = os.path.join(opt.saveroot, 'plots', filename))



if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TrainOptions().parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.method=='IPN':
        net = model.IPN(in_channels=opt.in_channels, channels=opt.channels, n_classes=opt.n_classes)
    if opt.method=='IPN_V2':
        net = model.IPN_V2(in_channels=opt.in_channels, channels=opt.channels, plane_perceptron_channels=opt.plane_perceptron_channels, n_classes=opt.n_classes, block_size=opt.block_size, plane_perceptron=opt.plane_perceptron)
    if(torch.cuda.is_available()):
        net=torch.nn.DataParallel(net,[0, 1]).cuda()
    #summary(net, (2,160,100,100), opt.batch_size)
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')

    #input the model into GPU
    #net.to(device=device)
    try:
        train_net(net=net,device=device)
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

