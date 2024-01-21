
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.layers import LeakyReLU, Lambda
import scipy.ndimage.morphology as morph

dw=4.0 # distance weight

def dice_coef(y_true, y_pred, smooth=1):
    intersection = keras.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.sum(y_true, axis=[1,2,3]) + keras.sum(y_pred, axis=[1,2,3])
    return keras.mean((2. * intersection + smooth) / (union + smooth),axis=0)

def weighted_binary_crossentropy(y_true, y_pred):
    wbce = -1.0*(100.0*y_true * keras.log(y_pred + keras.epsilon()) + (1-y_true) * keras.log(1-y_pred+keras.epsilon()))
    return keras.mean(wbce)




def distance_weighted_dice_coef(y_true,y_pred, smooth=1):
    dmaps = tf.map_fn( fn= lambda x: tf.cast(tf.py_function(morph.distance_transform_edt, [1.0-x[:,:,0]], Tout=np.float64),tf.float32), elems=y_true) #EDT online
    dmaps = keras.sigmoid(dmaps/6)
    dmaps= (dmaps - 0.5) * 2.0 + y_true[:,:,:,0] #rescale from 0.5 - 1.0 to 0.0 to 1.0 + weights 1.0 for gt mask
    w_pred = y_pred[:,:,:,0] * dmaps #weighted prediction
    #calculate non binary dice loss
    intersection = keras.sum(y_true[:,:,:,0] * w_pred, axis=[1,2])
    union = keras.sum(y_true[:,:,:,0], axis=[1,2]) + keras.sum(w_pred, axis=[1,2])
    return keras.mean((2. * intersection + smooth) / (union + smooth),axis=0)

def distance_weighted_dice_loss(y_true,y_pred, smooth=1):
    return 1-distance_weighted_dice_coef(y_true,y_pred,smooth)

def distance_weighted_bce_loss(y_true,y_pred, smooth=1):
    dmaps = tf.map_fn( fn= lambda x: tf.cast(tf.py_function(morph.distance_transform_edt, [1.0-x[:,:,0]], Tout=np.float64),tf.float32), elems=y_true) #EDT online
    dmaps = keras.sigmoid(dmaps/6)
    dmaps= (dmaps - 0.5) * 2.0 + y_true[:,:,:,0] #rescale from 0.5 - 1.0 to 0.0 to 1.0 + weights 1.0 for gt mask
    w_pred = y_pred[:,:,:,0] * dmaps #weighted prediction
    return keras.mean(keras.binary_crossentropy(y_true[:,:,:,0], w_pred), axis=0)

def dice_loss(y_true,y_pred):

    return 1 - dice_coef(y_true,y_pred)

def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def gating(g, channels):
    g1 = Conv2D(channels, 1, padding='same',kernel_initializer = 'he_normal')(g)
    g1 = Activation('relu') (g1)
    g1 = BatchNormalization()(g1)
    return g1

def AttentionBlock(x,g,F_int):
    shape_x = tf.keras.backend.int_shape(x)
    shape_g = tf.keras.backend.int_shape(g)

    #Wg
    g1 = Conv2D(F_int,1, padding = 'same', kernel_initializer = 'he_normal')(g)
    #Wx
    x1 = Conv2D(F_int,2, strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]),padding = 'same', kernel_initializer = 'he_normal')(x)
    gx = Add()([g1,x1])

    psi = Activation('relu')(gx)

    psi = Conv2D(1,1, padding = 'same', kernel_initializer = 'he_normal')(psi)
    psi = Activation('sigmoid') (psi)
    shape_sigmoid = tf.keras.backend.int_shape(psi)
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]), interpolation="bilinear")(psi)

    attn_coefficients = expend_as(upsample_sigmoid_xg, shape_x[3])
    out = Multiply()([attn_coefficients, x])
    return out


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_Enze19(pretrained_weights = None,input_size = (256,256,1)):
    # this model is based on the following paper by Enze Zhang et al.:
    # Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_Enze19_2(pretrained_weights = None,input_size = (512,512,1), distance_weight=4):
    # this model is based on the following paper by Enze Zhang et al.:
    # Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach
    global dw
    dw=distance_weight
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 128

    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 64
    
    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 32
    
    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 16
    
    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    
    up6 = Conv2DTranspose(256, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    merge6 = concatenate([conv4,up6], axis = 3)
    
    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2DTranspose(128, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    
    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2DTranspose(64, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    
    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = Conv2DTranspose(32, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    
    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    
    conv10 = Conv2D(1, 3, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    #compile
    if(dw!=0):
        model.compile(optimizer = Adam(lr = 1e-4), loss = distance_weighted_bce_loss, metrics = [BinaryCrossentropy(),distance_weighted_dice_coef])
    else:
        model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = [BinaryCrossentropy(),dice_coef])

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#our proposed Attention U-Net
def unet_Attention(pretrained_weights=None, input_size=(512, 512, 1), distance_weight=4):
    global dw
    dw=distance_weight
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

    conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

    conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

    conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
    conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

    conv5 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    conv5 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)

    up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4 = AttentionBlock(conv4,gating(conv5,256),128)
    merge6 = concatenate([conv4, up6], axis=3)

    conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    conv3= AttentionBlock(conv3,gating(conv6,128),64)
    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    conv2 = AttentionBlock(conv2,gating(conv7,64),32)
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    conv1 = AttentionBlock(conv1,gating(conv8,32),16)
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    # compile
    if(dw!=0):
        model.compile(optimizer = Adam(lr = 1e-4), loss = distance_weighted_bce_loss, metrics = [BinaryCrossentropy(),distance_weighted_dice_coef])
    else:
        model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = [BinaryCrossentropy(),dice_coef])


    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

