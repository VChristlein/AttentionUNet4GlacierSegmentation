import imageio
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from model import unet_Attention
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.layers import LeakyReLU, Lambda
import cv2
import argparse

def overlap(pred, gt, sar):
    green = (41 , 219 , 0)
    red = (234 , 0, 0)
    yellow = (255 , 228 , 0)

    imgg = cv2.cvtColor(sar,cv2.COLOR_GRAY2RGB)
    pred = pred[:,:,0] >  0.25

    intersec = gt * pred
    imgg[gt.astype(bool), 0] = green[0]
    imgg[gt.astype(bool), 1] = green[1]
    imgg[gt.astype(bool), 2] = green[2]

    imgg[pred.astype(bool), 0] = red[0]
    imgg[pred.astype(bool), 1] = red[1]
    imgg[pred.astype(bool), 2] = red[2]


    imgg[intersec.astype(bool), 0] = yellow[0]
    imgg[intersec.astype(bool), 1] = yellow[1]
    imgg[intersec.astype(bool), 2] = yellow[2]

    return imgg


def CLAHE(img):

    img = cv2.normalize(src=img, dst=None, alpha=0,
                  beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(50,50)).apply(img)
    return cv2.normalize(src=img, dst=None, alpha=0,
                  beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--modeldir', default='', type=str, help='Directory of the saved models', required=True)
parser.add_argument('--testimg', default='', type=str, help='Path to the SAR image', required=True)
parser.add_argument('--testimg_gt', default='', type=str, help='Path to the gt of the image', required=True)

args = parser.parse_args()
modelpath = args.modeldir

model = unet_Attention(input_size=(512,512,1),distance_weight=4)
#model.summary()






#img = imageio.imread("data_512_median_nopatch3/test/images/2007-01-01_RSAT_20_3.png")
img = imageio.imread(args.testimg)
#gt = imageio.imread("data_512_median_nopatch3/test/masks_lines/2007-01-01_RSAT_20_3_front.png")
gt = imageio.imread(args.testimg_gt)
sar = img.copy()
img = img/255



img = np.reshape(img,(1,512,512,1))

#epoch 5
model.load_weights(modelpath + 'weights00000005.h5')

modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda').output)
att4_5= modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_1').output)
att3_5 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_2').output)
att2_5 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_3').output)
att1_5 = modelAtt.predict(img)[0]
pred_5 = overlap(model.predict(img)[0],gt,sar)

#epoch 10
model.load_weights(modelpath + 'weights00000010.h5')
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda').output)
att4_10 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_1').output)
att3_10 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_2').output)
att2_10 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_3').output)
att1_10 = modelAtt.predict(img)[0]
pred_10 = overlap(model.predict(img)[0],gt,sar)

#epoch 50
model.load_weights(modelpath + 'weights00000050.h5')
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda').output)
att4_50 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_1').output)
att3_50 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_2').output)
att2_50 = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_3').output)
att1_50 = modelAtt.predict(img)[0]
pred_50 = overlap(model.predict(img)[0],gt,sar)

#epoch100
#model.load_weights(modelpath + 'weights00000100.h5')
#modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda').output)
#att4_100 = modelAtt.predict(img)[0]
#modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_1').output)
#att3_100 = modelAtt.predict(img)[0]
#modelAtt = Model(inputs=model.input, outputs = model.get_layer('activation_5').output)
#att2_100 = modelAtt.predict(img)[0]
#modelAtt = Model(inputs=model.input, outputs = model.get_layer('up_sampling2d_3').output)
#att1_100 = modelAtt.predict(img)[0]
#pred_100 = model.predict(img)[0]

#best epoch
model.load_weights(modelpath + 'model.hdf5')
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda').output)
att4_best = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_1').output)
att3_best = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_2').output)
att2_best = modelAtt.predict(img)[0]
modelAtt = Model(inputs=model.input, outputs = model.get_layer('lambda_3').output)
att1_best = modelAtt.predict(img)[0]
pred_best = overlap(model.predict(img)[0],gt,sar)

rows = 4 #=5 if you want to show epoch 100
cols = 5

fig, axis= plt.subplots(rows,cols)

for i in range(rows):
    for j in range(cols):
        axis[i, j].axis('off')

axis[0,0].imshow(att4_5[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[0, 0].set_title("attention_4")

axis[0,1].imshow(att3_5[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[0, 1].set_title("attention_3")
axis[0,2].imshow(att2_5[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[0, 2].set_title("attention_2")
axis[0,3].imshow(att1_5[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[0, 3].set_title("attention_1")
axis[0,4].imshow(pred_5)
axis[0, 4].set_title("prediction")
axis[1,0].imshow(att4_10[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[1,1].imshow(att3_10[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[1,2].imshow(att2_10[:,:,0],cmap='jet' , vmin=0, vmax=1)
axis[1,3].imshow(att1_10[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[1,4].imshow(pred_10)

axis[2,0].imshow(att4_50[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[2,1].imshow(att3_50[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[2,2].imshow(att2_50[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[2,3].imshow(att1_50[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[2,4].imshow(pred_50)

#axis[3,0].imshow(pred0_100[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[3,1].imshow(pred1_100[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[3,2].imshow(pred2_100[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[3,3].imshow(pred3_100[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[3,4].imshow(pred_gt_100[:,:,0],cmap='gray', vmin=0, vmax=1)
#axis[4,0].imshow(att4_best[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[4,1].imshow(att3_best[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[4,2].imshow(att2_best[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[4,3].imshow(att1_best[:,:,0],cmap='jet', vmin=0, vmax=1)
#axis[4,4].imshow(pred_best)

axis[3,0].imshow(att4_best[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[3,1].imshow(att3_best[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[3,2].imshow(att2_best[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[3,3].imshow(att1_best[:,:,0],cmap='jet', vmin=0, vmax=1)
axis[3,4].imshow(pred_best)


plt.show()