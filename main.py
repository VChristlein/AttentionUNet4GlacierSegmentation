from model import unet, unet_Enze19, unet_Enze19_2, unet_Attention
from data import trainGenerator, testGenerator, saveResult, saveResult_Amir
from clr_callback import CyclicLR
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, Callback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from scipy.spatial import distance
import argparse
import skimage.io as io

#%% Hyper-parameter tuning

parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--epochs', default=300, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('--batch_size', default=5, type=int, help='batch size (integer value)')
parser.add_argument('--patch_size', default=512, type=int, help='patch size (integer value)')

parser.add_argument('--early_stopping', action='store_true', help='If 1, classifier is using early stopping based on validation loss with patience 20 (0/1)')
parser.add_argument('--cyclic', action='store_true', help='If 1, a cyclic learning rate is used between 1e-5 and 1e-3')
parser.add_argument('--outpath', default='data_512_median_nopatch3', type=str, help='Output path for results and input path for data')
parser.add_argument('--distance_weight', default=4, type=int, help='distance weight (integer value), zero means no weighting')
parser.add_argument('--attention', action='store_true', help='Use attention mechanism')


args = parser.parse_args()


#%%
START=time.time()


PATCH_SIZE = args.patch_size
batch_size = args.batch_size

class WeightsSaver(Callback):
    def __init__(self, epochlist):
        self.epochlist = epochlist
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch in self.epochlist:
            name = str(Path(str(Out_Path),'weights%08d.h5' % self.epoch))
            self.model.save_weights(name)
        self.epoch+=1


num_samples = len([file for file in Path(args.outpath+'/train/images/').rglob('*.png')]) # number of training samples
num_val_samples = len([file for file in Path(args.outpath+'/val/images/').rglob('*.png')]) # number of validation samples

Out_Path = Path(args.outpath+'/test/masks_predicted_'+time.strftime("%y%m%d-%H%M%S")) # adding the time in the folder name helps to keep the results for multiple back to back exequations of the code


if not os.path.exists(Out_Path): os.makedirs(Out_Path)


data_gen_args = dict(horizontal_flip=True,
                    fill_mode='nearest')

train_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path(args.outpath+'/train')),
                        image_folder = 'images',
                        mask_folder = 'masks_lines',
                        aug_dict = None,
                        save_to_dir = None,
                        target_size=(PATCH_SIZE,PATCH_SIZE))

val_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path(args.outpath+'/val')),
                        image_folder = 'images',
                        mask_folder = 'masks_lines',
                        aug_dict = None, 
                        save_to_dir = None,
                        target_size=(PATCH_SIZE,PATCH_SIZE))

if(args.attention):
    model = unet_Attention(input_size=(PATCH_SIZE,PATCH_SIZE,1),distance_weight=args.distance_weight)
else:
    model = unet_Enze19_2(input_size=(PATCH_SIZE,PATCH_SIZE,1),distance_weight=args.distance_weight)
if(args.distance_weight==0):
    model_checkpoint = ModelCheckpoint(str(Path(str(Out_Path),'model.hdf5')), monitor='val_dice_coef', verbose=0, save_best_only=True, mode='max')
    early_stopping=keras.callbacks.EarlyStopping(patience=20, monitor = "val_dice_coef", mode ="max")
else:
    model_checkpoint = ModelCheckpoint(str(Path(str(Out_Path),'model.hdf5')), monitor='val_distance_weighted_dice_coef', verbose=0, save_best_only=True, mode='max')
    early_stopping=keras.callbacks.EarlyStopping(patience=20, monitor = "val_distance_weighted_dice_coef", mode ="max")
cyclic_lr = CyclicLR(
    mode ="triangular",
    base_lr=1e-5,
    max_lr=1e-3,
    step_size= 6* num_samples//batch_size)
ws = WeightsSaver([5,10,50,100])

callbacks = [model_checkpoint,ws]
if args.early_stopping:
    callbacks.append(early_stopping)
if args.cyclic:
    callbacks.append(cyclic_lr)
steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)
class_weight = {0: 0.000001, 1: 1.0}
History = model.fit_generator(train_Generator, 
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs,
                    validation_data=val_Generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks)

##########
##########
# save loss plot
plt.figure()
plt.rcParams.update({'font.size': 18})
plt.plot(model.history.epoch, model.history.history['loss'], 'X-', label='training loss', linewidth=3.0)
plt.plot(model.history.epoch, model.history.history['val_loss'], 'o-', label='validation loss', linewidth=3.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.tight_layout()
plt.grid(which='minor', linestyle='--')
plt.savefig(str(Path(str(Out_Path),'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)
#save diceplot
plt.figure()
plt.rcParams.update({'font.size': 18})
if args.distance_weight==0:
    plt.plot(model.history.epoch, model.history.history['val_dice_coef'],'o-', label="validation", linewidth=3.0)
    plt.plot(model.history.epoch, model.history.history['dice_coef'],'o-', label="training", linewidth=3.0)
else:
    plt.plot(model.history.epoch, model.history.history['val_distance_weighted_dice_coef'],'o-', label="validation", linewidth=3.0)
    plt.plot(model.history.epoch, model.history.history['distance_weighted_dice_coef'],'o-', label="training", linewidth=3.0)
plt.xlabel('epoch')
plt.ylabel('dice coefficient')
plt.legend(loc='lower right')
plt.minorticks_on()
plt.tight_layout()
plt.grid(which='minor', linestyle='--')
plt.savefig(str(Path(str(Out_Path),'dice_plot.png')), bbox_inches='tight', format='png', dpi=200)



END=time.time()
print('Execution Time: ', END-START)
