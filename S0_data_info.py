# -*- coding: utf-8 -*-

import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import os.path
import Amir_utils
import Michael_utils
import time
import cv2
import argparse
#%%


parser = argparse.ArgumentParser(description='Glacier Front Segmentation')


parser.add_argument('--patch_size', default=512, type=int, help='batch size (integer value)')
parser.add_argument('--outpath', default='data_512_median_nopatch', type=str, help='Output path for results')
parser.add_argument('--bilateral', action='store_true', help='Use bilateral filter for training/validation')
parser.add_argument('--clahe', action='store_true', help='Use CLAHE filter for training/validation')
parser.add_argument('--median', action='store_true', help='Use median filter for training/validation')
parser.add_argument('--padding', action='store_true', help='Use zero padding for testing data')
parser.add_argument('--thicken', action='store_true', help='Thicken the fronts')
parser.add_argument('--nopatch', action='store_true', help='Do not extract patches')
args= parser.parse_args()
out_path = args.outpath
PATCH_SIZE = args.patch_size
data_count = 0
data_all = []
data_names = []
data_augs = [0,1,2,3,4,5,6,7] #FLIP+ROT
for filename in sorted(Path('../front_detection/training-data-zone/').rglob('*.png')):



    data = imageio.imread(filename)
    data_all.append(data)
    
    data_names.append(filename)
    data_count += 1

print(data_count)


images_names, images_names_id = [], []
masks_line_names, masks_line_names_id = [], []
masks_zone_names, masks_zone_names_id = [], []
for i in range(len(data_names)):
    if "_front.png" in str(data_names[i]):
        masks_line_names.append(data_names[i])
        masks_line_names_id.append(i)
    elif "_zones.png" in str(data_names[i]):
        masks_zone_names.append(data_names[i])
        masks_zone_names_id.append(i)
    else:
        images_names.append(data_names[i])
        images_names_id.append(i)
        


#%% separate train and test images
from sklearn.model_selection import train_test_split

data_idx = np.arange(len(images_names_id))
train_idx, test_idx = train_test_split(data_idx, test_size=50, random_state=1) # 10 images are chosen as the test images
train_idx, val_idx = train_test_split(train_idx, test_size=50, random_state=1) # 50 images as validation data
        
#%% generate patches
START = time.time()
# ERROR: Some images are smaller than 512 and thus will be discarded with PATCH_SIZE=512 (We do not want it!)


#####
# train path
if not os.path.exists(str(Path(out_path+'/train/images'))): os.makedirs(str(Path(out_path+'/train/images')))
if not os.path.exists(str(Path(out_path+'/train/masks_zones'))): os.makedirs(str(Path(out_path+'/train/masks_zones')))
if not os.path.exists(str(Path(out_path+'/train/masks_lines'))): os.makedirs(str(Path(out_path+'/train/masks_lines')))

STRIDE_train = (PATCH_SIZE,PATCH_SIZE)
patch_counter_train = 0
for i in train_idx:

    blurrfactor = int(np.ceil(50/int(str(images_names[i]).split("_")[-2]) ))
    if(blurrfactor%2==0): blurrfactor+=1
    print(str(images_names[i]) + " " + str(blurrfactor))
    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp==127] = 0
    masks_zone_tmp[masks_zone_tmp==254] = 255
    img = data_all[images_names_id[i]]
    mask_line = data_all[masks_line_names_id[i]]
    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement
    if args.bilateral:
        img = cv2.bilateralFilter(img,20,80,80)
    if args.clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25)).apply(img)
    if args.median:

        img = cv2.medianBlur(img,blurrfactor)


    img_pad = img
    mask_zone_pad = masks_zone_tmp
    mask_line_pad = mask_line


    if args.thicken==True:
        kernel_size= int(np.max(img.shape)/ PATCH_SIZE * 6)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_line_pad = cv2.dilate(mask_line_pad,kernel,iterations = 1)


    if args.nopatch ==True:
       s = img_pad.shape
       if s[0]>= s[1]:
           s=(PATCH_SIZE,int(s[1]*PATCH_SIZE/s[0]))
       else:
           s=(int(s[0]*PATCH_SIZE/s[1]),PATCH_SIZE)
       s=s[::-1]
       img_pad = cv2.resize(img_pad,dsize=s,interpolation=cv2.INTER_LINEAR)
       masks_zone_tmp = cv2.resize(masks_zone_tmp,dsize=s,interpolation=cv2.INTER_NEAREST)
       mask_line = cv2.resize(mask_line_pad,dsize=s,interpolation=cv2.INTER_NEAREST)

    #padding
    if args.padding==True:
        img_pad = cv2.copyMakeBorder(img_pad, 0, (PATCH_SIZE - img_pad.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - img_pad.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        mask_zone_pad = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        mask_line_pad = cv2.copyMakeBorder(mask_line, 0, (PATCH_SIZE - mask_line.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - mask_line.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)


    p_masks_zone = [mask_zone_pad]
    p_img = [img_pad]
    p_masks_line=[mask_line_pad]

    if args.nopatch==False:
        p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(mask_zone_pad, (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_train)
        p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_train)
        p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(mask_line_pad, (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_train)

    for j in range(len(p_masks_zone)):

        for k in data_augs: #8 data augmentation modes
            masks_zone_aug = Michael_utils.data_augmentation(p_masks_zone[j], k)
            img_aug = Michael_utils.data_augmentation(p_img[j], k)
            masks_line_aug = Michael_utils.data_augmentation(p_masks_line[j], k)
            cv2.imwrite(str(Path(out_path+'/train/images/'+str(patch_counter_train)+'_'+str(k)+'.png')), img_aug)
            cv2.imwrite(str(Path(out_path+'/train/masks_zones/'+str(patch_counter_train)+'_'+str(k)+'.png')), masks_zone_aug)
            cv2.imwrite(str(Path(out_path+'/train/masks_lines/'+str(patch_counter_train)+'_'+str(k)+'.png')), masks_line_aug)
        patch_counter_train += 1





#####
# validation path
if not os.path.exists(str(Path(out_path+'/val/images'))): os.makedirs(str(Path(out_path+'/val/images')))
if not os.path.exists(str(Path(out_path+'/val/masks_zones'))): os.makedirs(str(Path(out_path+'/val/masks_zones')))
if not os.path.exists(str(Path(out_path+'/val/masks_lines'))): os.makedirs(str(Path(out_path+'/val/masks_lines')))

STRIDE_val = (PATCH_SIZE,PATCH_SIZE)
patch_counter_val = 0
for i in val_idx:
    blurrfactor = int(np.ceil(50 / int(str(images_names[i]).split("_")[-2])))
    if (blurrfactor % 2 == 0): blurrfactor += 1
    print(str(images_names[i]) + " " + str(blurrfactor))

    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp==127] = 0
    masks_zone_tmp[masks_zone_tmp==254] = 255
    
    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement
    img = data_all[images_names_id[i]]
    mask_line = data_all[masks_line_names_id[i]]
    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement
    if args.bilateral:
        img = cv2.bilateralFilter(img, 20, 80, 80)
    if args.clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply(img)
    if args.median:
        img = cv2.medianBlur(img, blurrfactor)
    #padding
    img_pad = img
    mask_zone_pad = masks_zone_tmp
    mask_line_pad = mask_line

    if args.thicken==True:
        kernel_size= int(np.max(img.shape)/ PATCH_SIZE * 6)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_line_pad = cv2.dilate(mask_line_pad,kernel,iterations = 1)

    if args.nopatch ==True:
       s = img_pad.shape
       if s[0]>= s[1]:
           s=(PATCH_SIZE,int(s[1]*PATCH_SIZE/s[0]))
       else:
           s=(int(s[0]*PATCH_SIZE/s[1]),PATCH_SIZE)
       s=s[::-1]
       img_pad = cv2.resize(img_pad,dsize=s,interpolation=cv2.INTER_LINEAR)
       masks_zone_tmp = cv2.resize(masks_zone_tmp,dsize=s,interpolation=cv2.INTER_NEAREST)
       mask_line = cv2.resize(mask_line_pad,dsize=s,interpolation=cv2.INTER_NEAREST)



    if args.padding==True:
        img_pad = cv2.copyMakeBorder(img_pad, 0, (PATCH_SIZE - img_pad.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - img_pad.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        mask_zone_pad = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        mask_line_pad = cv2.copyMakeBorder(mask_line, 0, (PATCH_SIZE - mask_line.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - mask_line.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)


    p_masks_zone = [mask_zone_pad]
    p_img = [img_pad]
    p_masks_line = [mask_line_pad]

    if args.nopatch == False:
        p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(mask_zone_pad, (PATCH_SIZE, PATCH_SIZE),
                                                                          stride=STRIDE_val)
        p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)
        p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(mask_line_pad, (PATCH_SIZE, PATCH_SIZE),
                                                                          stride=STRIDE_val)

    for j in range(len(p_masks_zone)):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        #if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 1:
        for k in [0,1,2,3,4,5,6,7]: #data augmentation
            masks_zone_aug = Michael_utils.data_augmentation(p_masks_zone[j], k)
            img_aug = Michael_utils.data_augmentation(p_img[j], k)
            masks_line_aug = Michael_utils.data_augmentation(p_masks_line[j], k)
            cv2.imwrite(str(Path(out_path+'/val/images/'+str(patch_counter_val)+'_'+str(k)+'.png')), img_aug)
            cv2.imwrite(str(Path(out_path+'/val/masks_zones/'+str(patch_counter_val)+'_'+str(k)+'.png')), masks_zone_aug)
            cv2.imwrite(str(Path(out_path+'/val/masks_lines/'+str(patch_counter_val)+'_'+str(k)+'.png')), masks_line_aug)
        patch_counter_val += 1




#####
# test path
if not os.path.exists(str(Path(out_path+'/test/images'))): os.makedirs(str(Path(out_path+'/test/images')))
if not os.path.exists(str(Path(out_path+'/test/masks_zones'))): os.makedirs(str(Path(out_path+'/test/masks_zones')))
if not os.path.exists(str(Path(out_path+'/test/masks_lines'))): os.makedirs(str(Path(out_path+'/test/masks_lines')))

STRIDE_test = (PATCH_SIZE,PATCH_SIZE)
patch_counter_test = 0
for i in test_idx:
    blurrfactor = int(np.ceil(50 / int(str(images_names[i]).split("_")[-2])))
    if (blurrfactor % 2 == 0): blurrfactor += 1
    print(str(images_names[i]) + " " + str(blurrfactor))

    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp==127] = 0
    masks_zone_tmp[masks_zone_tmp==254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement
    img = data_all[images_names_id[i]]
    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement
    if args.bilateral:
        img = cv2.bilateralFilter(img, 20, 80, 80)
    if args.clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply(img)
    if args.median:
        img = cv2.medianBlur(img, blurrfactor)


    mask_line = data_all[masks_line_names_id[i]]

    if args.thicken==True:
        kernel_size= int(np.max(img.shape)/ PATCH_SIZE * 6)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_line = cv2.dilate(mask_line,kernel,iterations = 1)


    if args.nopatch ==True:
       s = img.shape
       if s[0]>= s[1]:
           s=(PATCH_SIZE,int(s[1]*PATCH_SIZE/s[0]))
       else:
           s=(int(s[0]*PATCH_SIZE/s[1]),PATCH_SIZE)
       s=s[::-1]
       img = cv2.resize(img,dsize=s,interpolation=cv2.INTER_LINEAR)
       masks_zone_tmp = cv2.resize(masks_zone_tmp,dsize=s,interpolation=cv2.INTER_NEAREST)
       mask_line = cv2.resize(mask_line,dsize=s,interpolation=cv2.INTER_NEAREST)


    if args.padding==True:
        img = cv2.copyMakeBorder(img, 0, (PATCH_SIZE - img.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        masks_zone_tmp = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)
        mask_line = cv2.copyMakeBorder(mask_line, 0, (PATCH_SIZE - mask_line.shape[0]) % PATCH_SIZE, 0,
                                    (PATCH_SIZE - mask_line.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT,0)


    cv2.imwrite(str(Path(out_path+'/test/images/'+Path(images_names[i]).name)), img)
    cv2.imwrite(str(Path(out_path+'/test/masks_zones/'+Path(masks_zone_names[i]).name)), masks_zone_tmp)
    cv2.imwrite(str(Path(out_path+'/test/masks_lines/'+Path(masks_line_names[i]).name)), mask_line)
    


END = time.time()
print(END-START) 
