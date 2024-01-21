import skimage.io as io
import Amir_utils
import numpy as np
from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt
from model import unet, unet_Enze19, unet_Enze19_2, unet_Attention
from data import trainGenerator, testGenerator, saveResult, saveResult_Amir
import keras
import scipy.ndimage.morphology as morph
from keras.models import load_model
from scipy.spatial import distance
import os
import time
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')
import tensorflow as tf

parser.add_argument('--patch_size', default=512, type=int, help='batch size (integer value)')
parser.add_argument('--modelpath', default='', type=str, help='Input path for model', required=True)
parser.add_argument('--datapath', default='data_512_median_nopatch/test', type=str, help='Input path for test data', required=True)
parser.add_argument('--outpath', default='data_512_median_nopatch/test/masks_predicted_'+time.strftime("%y%m%d-%H%M%S"), type=str, help='Output path for results')
parser.add_argument('--padding', action='store_true', help='Use zero padding for testing data')
parser.add_argument('--nopatch', action='store_true', help='Do not use patches')
parser.add_argument('--lines', action='store_true', help='Use fronts as gt instead of zones')
parser.add_argument('--nothreshold', action='store_true', help='Do not search for the optimal binary threshold')
parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
parser.add_argument('--distance_weight', default=0, type=int, help='Distance weight the model trained on (0/4/8/16)')

args = parser.parse_args()

if not (os.path.exists(args.modelpath) and os.path.exists(args.datapath)):
    print("Model path or data path invalid")
    exit(1)


PATCH_SIZE = args.patch_size


test_path = args.datapath


Out_Path = args.outpath # adding the time in the folder name helps to keep the results for multiple back to back exequations of the code
if not os.path.exists(Out_Path): os.makedirs(Out_Path)
if args.attention:
    model = unet_Attention(input_size=(PATCH_SIZE,PATCH_SIZE,1),distance_weight=args.distance_weight)
else:
    model= unet_Enze19_2(input_size=(PATCH_SIZE,PATCH_SIZE,1),distance_weight=args.distance_weight)
model.load_weights(args.modelpath)
#model.summary()

thresholds = np.arange(0.0,1.0,0.01) #0.85
if(args.nothreshold==True):
    thresholds=[0.5]
DICE_all = [[] for i in range(len(thresholds))]
EUCL_all = [[] for i in range(len(thresholds))]
IoU_all = [[] for i in range(len(thresholds))]
WDICE8_all = [[] for i in range(len(thresholds))]
WDICE16_all = [[] for i in range(len(thresholds))]
WDICE4_all = [[] for i in range(len(thresholds))]
DICE_avg = []
EUCL_avg = []
IoU_avg = []
WDICE8_avg = []
WDICE4_avg = []
WDICE16_avg = []

predictions = []
test_file_names = []
Perf = {}


def distance_weighted_dice_coef(y_true,y_pred, weight=8.0,smooth=1 ):

    dist = morph.distance_transform_edt(1.0-y_true)/weight
    dist = 1/(1 + np.exp(-1*dist))
    dist = (dist -0.5) * 2.0 + y_true
    wp = dist * y_pred
    dice = (2* np.sum(wp * y_true) + smooth) / (np.sum(y_true) + np.sum(wp)+smooth)
    return dice


for filename in Path(test_path, 'images').rglob('*.png'):

    print(filename.name)
    test_file_names.append(filename.name)
    img = io.imread(filename, as_gray=True)
    img = img / 255
    img_pad=img
    if args.padding==True:
        img_pad = cv2.copyMakeBorder(img, 0, (PATCH_SIZE - img.shape[0]) % PATCH_SIZE, 0,
                                (PATCH_SIZE - img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

    img_mask_predicted_recons = img_pad
    if args.nopatch==False:
        p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE, PATCH_SIZE),
                                                        stride=(PATCH_SIZE, PATCH_SIZE))



        p_img = np.reshape(p_img, p_img.shape + (1,))

        p_img_predicted = model.predict(p_img)


        p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])


        img_mask_predicted_recons = Amir_utils.reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]
    else:
        p_img = np.reshape(img_pad,(1,)+img_pad.shape + (1,))

        p_img_predicted = model.predict(p_img)
        p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])
        img_mask_predicted_recons = p_img_predicted[0]




    # unpad and normalize
    img_mask_predicted_recons_unpad = img_mask_predicted_recons[0:img.shape[0], 0:img.shape[1]]
    img_mask_predicted_recons_unpad_norm = cv2.normalize(src=img_mask_predicted_recons_unpad, dst=None, alpha=0,
                                                         beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    predictions.append(img_mask_predicted_recons_unpad_norm)

    gt_path = str(Path(test_path, 'masks_zones'))
    gt_name = filename.name.partition('.')[0] + '_zones.png'

    if(args.lines==True):
        gt_path = str(Path(test_path, 'masks_lines'))
        gt_name = filename.name.partition('.')[0] + '_front.png'

    gt = io.imread(str(Path(gt_path, gt_name)), as_gray=True)
    for i in range(len(thresholds)):
        # quantization to make the binary masks
        img_tmp = img_mask_predicted_recons_unpad_norm.copy()
        if (args.nothreshold == False):
            img_tmp[img_tmp < thresholds[i]*255] = 0
            img_tmp[img_tmp >= thresholds[i]*255] = 255

        intersection = np.sum((gt.flatten()/255.0) * (img_tmp.flatten()/255.0))

        union = (np.sum(gt.flatten()/255.0) + np.sum(img_tmp.flatten()/255.0) )
        if union==0:
            dice = 1
            DICE_all[i].append(dice)
        else:
            dice = (2* intersection +1.0)/ (union+1.0)
            DICE_all[i].append(dice)

        IoU_all[i].append(np.sum(intersection) / (np.sum(union)-intersection))
        EUCL_all[i].append(distance.euclidean(gt.flatten()/255, img_tmp.flatten()/255))
        WDICE8_all[i].append(distance_weighted_dice_coef(gt/255,img_tmp/255,8.0))

        WDICE4_all[i].append(distance_weighted_dice_coef(gt/255,img_tmp/255,4.0))

        WDICE16_all[i].append(distance_weighted_dice_coef(gt/255,img_tmp/255,16.0))

#calculate the mean

for i in range(len(thresholds)):
    WDICE8_avg.append(np.mean(WDICE8_all[i]))
    WDICE4_avg.append(np.mean(WDICE4_all[i]))
    WDICE16_avg.append(np.mean(WDICE16_all[i]))
    DICE_avg.append(np.mean(DICE_all[i]))
    EUCL_avg.append(np.mean(EUCL_all[i]))
    IoU_avg.append(np.mean(IoU_all[i]))

WDICE8_argmax = np.argmax(WDICE8_avg)
WDICE4_argmax = np.argmax(WDICE4_avg)
WDICE16_argmax = np.argmax(WDICE16_avg)
DICE_argmax = np.argmax(DICE_avg)
EUCL_argmax = np.argmin(EUCL_avg)
IoU_argmax = np.argmax(IoU_avg)



#save the predicted images with best threshold
for i in range(len(predictions)):
    img = predictions[i]
    if (args.nothreshold == False):
        if(args.distance_weight==0):
            img[img < thresholds[DICE_argmax]*255] = 0
            img[img >= thresholds[DICE_argmax]*255] =255
        if(args.distance_weight==4):
            img[img < thresholds[WDICE4_argmax]*255] = 0
            img[img >= thresholds[WDICE4_argmax]*255] =255
        if(args.distance_weight==8):
            img[img < thresholds[WDICE8_argmax]*255] = 0
            img[img >= thresholds[WDICE8_argmax]*255] =255
        if(args.distance_weight==16):
            img[img < thresholds[WDICE16_argmax]*255] = 0
            img[img >= thresholds[WDICE16_argmax]*255] =255
    io.imsave(Path(str(Out_Path), test_file_names[i]), img)

plt.figure(figsize=(16,6))
plt.rcParams.update({'font.size': 18})
plt.subplot(1,2,1)
plt.plot(thresholds, WDICE8_avg, label='WDICE_8', linewidth=2.0)
plt.plot(thresholds, WDICE4_avg, label='WDICE_4', linewidth=2.0)
plt.plot(thresholds, WDICE16_avg, label='WDICE_16', linewidth=2.0)
plt.plot(thresholds, DICE_avg, label='DICE', linewidth=2.0)
plt.xlabel('threshold')
#plt.plot(thresholds, IoU_avg, label='IoU', linewidth=2.0)
plt.legend(loc='lower right')

plt.subplot(1,2,2)

plt.plot(thresholds, EUCL_avg, label='EUCL', linewidth=2.0)
plt.xlabel('threshold')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.grid(which='minor', linestyle='--')
plt.savefig(str(Path(str(Out_Path),'threshold.png')), bbox_inches='tight', format='png', dpi=200)



#save the images with the best IoU Coefficient


Perf['DICE_all'] = DICE_all
Perf['DICE_avg'] = DICE_avg
Perf['EUCL_all'] = EUCL_all
Perf['EUCL_avg'] = EUCL_avg
Perf['WDICE8_all'] = WDICE8_all
Perf['WDICE8_avg'] = WDICE8_avg
Perf['WDICE4_all'] = WDICE4_all
Perf['WDICE4_avg'] = WDICE4_avg
Perf['WDICE16_all'] = WDICE16_all
Perf['WDICE16_avg'] = WDICE16_avg
Perf['test_file_names'] = test_file_names
np.savez(str(Path(str(Out_Path), 'Performance.npz')), Perf)

with open(str(Path(str(Out_Path), 'ReportOnModel.txt')), 'a') as f:
    f.write('best DICE: ' + str(DICE_avg[DICE_argmax]) + '\t threshold: ' + str(thresholds[DICE_argmax]) + '\n' +
            'best WDICE4: ' + str(WDICE4_avg[WDICE4_argmax]) + '\t threshold: ' + str(thresholds[WDICE4_argmax]) + '\n' +
            'best WDICE8: ' + str(WDICE8_avg[WDICE8_argmax]) + '\t threshold: ' + str(thresholds[WDICE8_argmax]) + '\n' +
            'best WDICE16: ' + str(WDICE16_avg[WDICE16_argmax]) + '\t threshold: ' + str(thresholds[WDICE16_argmax]) + '\n' +
            'best EUCL: ' + str(EUCL_avg[EUCL_argmax]) + '\t threshold: ' + str(thresholds[EUCL_argmax]) + '\n' +
            'best IoU: ' + str(IoU_avg[IoU_argmax]) + '\t threshold: ' + str(thresholds[IoU_argmax]) )
