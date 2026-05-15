import torch
import torch.nn as nn
import logging
import sys
import os
import model
import numpy as np
import scipy.misc as misc
from options.test_options import TestOptions
import natsort
from scipy import io
from skimage.transform import resize
import imageio.v2 as imageio
from PIL import Image
from matplotlib import pyplot as plt
import  matplotlib

from plot_reliability_diagram import *



def test_net(net,device):
    DATA_SIZE = opt.data_size
    BLOCK_SIZE = opt.block_size
    test_results = os.path.join(opt.saveroot, 'test_results')
    feature_results= opt.feature_dir
    net.eval()
    test_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]))
    cube_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))

    modalitylist = opt.modality_filename
    logging.info(modalitylist)
    # modalitylist=['image1', 'image1', 'label']
    
    testids = opt.test_ids
    valids = opt.val_ids
    trainids= opt.train_ids
    cubelist0 = os.listdir(os.path.join(opt.dataroot, 'test', modalitylist[0]))
    cubelist0 = natsort.natsorted(cubelist0)
    cubelist =cubelist0[trainids[0]:trainids[1]]+cubelist0[valids[0]:valids[1]]+cubelist0[testids[0]:testids[1]]
    #cubelist = cubelist0[valids[0]:valids[1]] + cubelist0[testids[0]:testids[1]]

    vote_time=4
    for kk,cube in enumerate(cubelist): # modify the length
        bscanlist = os.listdir(os.path.join(opt.dataroot, 'test', modalitylist[0], cube))
        bscanlist=natsort.natsorted(bscanlist)
        for i,bscan in enumerate(bscanlist):
            for j,modal in enumerate(modalitylist):
                if modal!=opt.modality_filename[-1]:
                    #cube_images[0,j,:,:,i]=np.array(misc.imresize(misc.imread(os.path.join(opt.dataroot, 'test', modal,cube,bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
                    cube_images[0,j,:,:,i]=np.array(resize(imageio.imread(os.path.join(opt.dataroot, 'test', modal, cube, bscan))/255.0, (BLOCK_SIZE[0], DATA_SIZE[1]), order=1))
        #logging.info(cube_images[0, 0,:,:,:])
        result =np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
        featuremap=np.zeros((opt.plane_perceptron_channels,DATA_SIZE[1], DATA_SIZE[2]))
        votemap=np.zeros((DATA_SIZE[1], DATA_SIZE[2]))

        for i in range(0,DATA_SIZE[1]-BLOCK_SIZE[1]+BLOCK_SIZE[1]//vote_time,BLOCK_SIZE[1]//vote_time):
            for j in range(0,DATA_SIZE[2]-BLOCK_SIZE[2]+BLOCK_SIZE[2]//vote_time,BLOCK_SIZE[2]//vote_time):
                test_images[0, :, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = cube_images[0, :, :,i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]
                images = torch.from_numpy(test_images)
                images = images.to(device=device, dtype=torch.float32)
                #images = images.cuda()
                #net = net.double()
                pred,features = net(images)
                pred = torch.nn.functional.softmax(pred, dim=1)
                
                #confidences = np.max(pred, dim=1)
                #predictions = np.argmax(pred, dim=1)
                
                votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]=votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+1
                result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+pred[0,1,0,:,:].cpu().detach().numpy()
                featuremap[:,i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]] = featuremap[:,i:i + BLOCK_SIZE[1],j:j + BLOCK_SIZE[2]] + features[0,:,:,:].cpu().detach().numpy()

        modified_result = result/votemap
        modified_result = modified_result.reshape(160000,1)
        y_pred = np.zeros((160000, 2)) # 0 is black, 1 is white
        y_pred[:, 0] = 1 - modified_result[:, 0] # the probability of being black
        y_pred[:, 1] = modified_result[:, 0] # the probability of being white

        
        result=result/votemap*255 # grayscale
        featuremap=featuremap/votemap
        print(cube)
        
        filename = os.environ["SLURM_JOB_ID"] + "_" + cube
        label_arr = imageio.imread(os.path.join(opt.dataroot, 'test', 'label', cube + ".bmp"))
        label_arr[label_arr==100] = 0
        label_arr[label_arr>=1] = 1
        label_arr = np.rot90(label_arr, 3) # rotate three times = rotate clockwise once
        #   y_true classes, m*1
        #   m = number of samples 400*400
        #   the numbers correspond to the class IDs
        
        #   y_pred load the numpy array with the probability matrices m X n
        #   m = number of samples
        #   n = number of classes
        #   confidence/prob per pixel = np.max(y_pred[row_idx, :])
        
        
        # heatmap
        # am I showing confidence or uncertainty?
        # Choose a colormap (for example, 'viridis')
        cmap = matplotlib.colormaps['seismic']
        
        # plt.colorbar(label='Probability/Confidence')
        # plt.title('Probability Heatmap')
        heatmap = cmap(result/255)*255
        imageio.imwrite(os.path.join(opt.saveroot, 'heatmap', filename + "_heatmap" + ".bmp"), heatmap.astype(np.uint8))
       
        # Reliability Diagram
        y_true = label_arr.reshape(160000, 1)
        plot_reliabiblity_diagram(y_true=y_true, y_pred=y_pred, n_bins=20, rel_diag_folder= os.path.join(opt.saveroot, 'reliability_diagrams', filename))
        
        result[result>127.5] = 255.0
        result[result<=127.5] = 0.0
        difference = np.abs(result - label_arr*255)
        imageio.imwrite(os.path.join(opt.saveroot, 'plots', filename + "_diff" + ".bmp"), difference.astype(np.uint8))

        
        #misc.imsave(os.path.join(test_results, cube + ".bmp"), result.astype(np.uint8))

        imageio.imwrite(os.path.join(test_results, filename + cube + ".bmp"), result.astype(np.uint8))
        np.save(os.path.join(feature_results, filename + cube + ".npy"),featuremap)
        #io.savemat(os.path.join(feature_results, cube + ".mat"), {'feature':featuremap})
if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.method == 'IPN':
        net = model.IPN(in_channels=opt.in_channels, channels=opt.channels, n_classes=opt.n_classes)
    if opt.method == 'IPN_V2':
        net = model.IPN_V2(in_channels=opt.in_channels, channels=opt.channels,plane_perceptron_channels=opt.plane_perceptron_channels, n_classes=opt.n_classes,
                           block_size=opt.block_size, plane_perceptron=opt.plane_perceptron)

    #load trained model
    bestmodelpath= os.path.join(opt.saveroot, 'best_model', natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(opt.saveroot, 'best_model', natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])+'/'+os.listdir(bestmodelpath)[0]
    print(restore_path)
    #restore_path = os.path.join(opt.saveroot, 'checkpoints', '27000.pth')
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
