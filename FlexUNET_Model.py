# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:59:56 2022

@author: Ashkan
"""

import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Lambda, 
                                            UpSampling2D, Activation, 
                                            BatchNormalization, MaxPooling2D,
                                            concatenate, Add, Concatenate, 
                                            MaxPool2D, Dropout, LeakyReLU)
from tensorflow.keras.regularizers import l2
# from tensorflow.python.keras.models import Model

from tensorflow.keras.models import Model
from tensorflow import keras
from keras_unet_collection._backbone_zoo import bach_norm_checker, backbone_zoo
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right
from keras_unet_collection.layer_utils import CONV_output, decode_layer, CONV_stack, encode_layer

import warnings

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 256, 3]
inputs_size = input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

## Flex-U-Net

def Con_Exp_Path(X, Y, n):
    L = Conv2D(2 ** X, (3, 3), kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(n)
    L = BatchNormalization()(L)
    L = Activation('relu')(L)
    L = Conv2D(2 ** X, (3, 3), kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(L)
    L = BatchNormalization()(L)
    L = Activation('relu')(L)
    L = Dropout(Y)(L)
    L = Conv2D(2 ** X, (3, 3), activation='relu', kernel_initializer='he_normal',  # 'glorot_uniform', 'he_normal'
                padding='same')(L)
    return L

##VGGNET
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def Flex_U_Net_Segmentation(NN, KK, dd, Final_Act, backbone=None):    
    if backbone==None:
        NL = 9 + (KK + dd - 1) * 5  # No. of layers for EU-Net architecture
        MD = [None] * (NL)  # EU_Net Layers combination
        MD[0] = Input(input_size)
        # MD[1] = Lambda(lambda x: x/255)(MD[0])
        MD
        for tt in np.arange(0, KK):
            MD[2 * (tt + 1)] = Con_Exp_Path(NN + tt, 0.1 * (1 + np.fix(tt / 2)), MD[2 * (tt) + 1])
            MD[2 * (tt + 1) + 1] = MaxPooling2D((2, 2))(MD[2 * (tt + 1)])
    
        gg = 2 * (KK + 1)  # e.g. for KK=3 & dd=1==>gg=8 or for KK=2 & dd=2==>gg=6
        MD[gg] = Con_Exp_Path(NN + KK, 0.1 * (1 + np.fix((tt + 1) / 2)), MD[gg - 1])
    
        for tt in np.arange(dd - 1, -1, -1):
            MD[gg + 3 * (dd - tt - 1) + 1] = Conv2DTranspose(2 ** (NN + KK - (dd - tt)), \
                                                              (2, 2), strides=(2, 2), \
                                                              padding='same')(MD[gg + 3 * (dd - tt - 1)])
            MD[gg + 3 * (dd - tt - 1) + 2] = concatenate([MD[gg + 3 * (dd - tt - 1) + 1], MD[gg - 2 * (dd - tt)]])
            MD[gg + 3 * (dd - tt - 1) + 3] = Con_Exp_Path(NN + KK - (dd - tt), 0.1 * (1 + np.fix((KK - dd + tt) / 2)), \
                                                          MD[gg + 3 * (dd - tt - 1) + 2])
    
        gg += 3 * dd  # e.g. for KK=3 & dd=1 ==> gg=11 or for KK=2 & dd=2==>gg=12
    
        for tt in np.arange(0, dd):
            MD[gg + 2 * tt + 1] = MaxPooling2D((2, 2))(MD[gg + 2 * tt])
            MD[gg + 2 * tt + 2] = Con_Exp_Path(NN + KK + (tt + 1 - dd), 0.1 * (1 + np.fix((KK - dd + tt + 1) / 2)), \
                                                MD[gg + 2 * tt + 1])
    
        gg += 2 * dd  # e.g. for KK=3 & dd=1 ==> gg=13 or for KK=2 & dd=2==>gg=16
    
        for tt in np.arange(dd - 1, -1, -1):
            MD[gg + 3 * (dd - tt - 1) + 1] = Conv2DTranspose(2 ** (NN + KK - (dd - tt)), (2, 2), strides=(2, 2), \
                                                              padding='same')(MD[gg + 3 * (dd - tt - 1)])
            MD[gg + 3 * (dd - tt - 1) + 2] = concatenate([MD[gg + 3 * (dd - tt - 1) + 1], MD[gg - 2 * (dd - tt)]])
            MD[gg + 3 * (dd - tt - 1) + 3] = Con_Exp_Path(NN + KK - (dd - tt), 0.1 * (1 + np.fix((KK - dd + tt) / 2)), \
                                                          MD[gg + 3 * (dd - tt - 1) + 2])
    
        gg += 3 * dd  # e.g. for KK=3 & dd=1 ==> gg=16 or for KK=2 & dd=2==>gg=22
    
        for tt in np.arange(KK, dd, -1):
            MD[gg + 3 * (KK - tt) + 1] = Conv2DTranspose(2 ** (NN + tt - (dd + 1)), (2, 2), strides=(2, 2), \
                                                          padding='same')(MD[gg + 3 * (KK - tt)])
            MD[gg + 3 * (KK - tt) + 2] = concatenate([MD[gg + 3 * (KK - tt) + 1], MD[2 * (tt - dd)]])
            MD[gg + 3 * (KK - tt) + 3] = Con_Exp_Path(NN + tt - (dd + 1), 0.1 * (1 + np.fix((tt - dd - 1) / 2)), \
                                                      MD[gg + 3 * (KK - tt) + 2])
        if gg != NL - 1:
            gg += 3 * (KK - dd)
    
        # MD[gg+1] = Conv2D(1,(1,1), activation='sigmoid')(MD[gg])
        # MD[gg + 1] = Conv2D(1, (1, 1))(MD[gg])
        # MD[gg + 1] = Activation(Final_Act)(MD[gg + 1])
        if Final_Act == 'selu':
            MD[gg + 1]  = Conv2D(1, kernel_size=(1, 1), activation=Final_Act, 
                            kernel_initializer='lecun_normal')(MD[gg])
        else:
            MD[gg + 1] = Conv2D(1, kernel_size=(1, 1), activation=Final_Act)(MD[gg])
    
        model = Model(inputs=[MD[0]], outputs=[MD[gg + 1]], name='EU-Net')
    elif 'vgg' in backbone:
        if KK > 4:
            KK=4
        if dd >4:
            dd=4
        if dd==0:
            dd=1
        if backbone =='vgg16':
            inputs = Input(input_size)
            vgg16 = keras.applications.VGG16(include_top=False, 
                                             weights="imagenet", 
                                             input_tensor=inputs)
            xx = Lambda(lambda x: x/255)(inputs)
            if KK==4 and dd==4:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1)
                c2=vgg16.layers[3](c1)
                c2=vgg16.layers[4](c2)
                c2=vgg16.layers[5](c2)
                c3=vgg16.layers[6](c2)
                c3=vgg16.layers[7](c3)
                c3=vgg16.layers[8](c3)
                c3=vgg16.layers[9](c3)
                c4=vgg16.layers[10](c3)
                c4=vgg16.layers[11](c4)
                c4=vgg16.layers[12](c4)
                c4=vgg16.layers[13](c4)
                c5=vgg16.layers[14](c4)
                c5=vgg16.layers[15](c5)
                c5=vgg16.layers[16](c5)
                c5=vgg16.layers[17](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)
                se3=decoder_block(se2,c2,128)
                se4=decoder_block(se3,c1,64)
                #### Sub-Contracting Path
                sc1=vgg16.layers[3](se4)
                sc1=vgg16.layers[4](sc1)
                sc1=vgg16.layers[5](sc1)
                sc2=vgg16.layers[6](sc1)
                sc2=vgg16.layers[7](sc2)
                sc2=vgg16.layers[8](sc2)
                sc2=vgg16.layers[9](sc2)
                sc3=vgg16.layers[10](sc2)
                sc3=vgg16.layers[11](sc3)
                sc3=vgg16.layers[12](sc3)
                sc3=vgg16.layers[13](sc3)
                sc4=vgg16.layers[14](sc3)
                sc4=vgg16.layers[15](sc4)
                sc4=vgg16.layers[16](sc4)
                sc4=vgg16.layers[17](sc4)
                #### Expanding Path
                e1=decoder_block(sc4,sc3,512)
                e2=decoder_block(e1,sc2,256)
                e3=decoder_block(e2,sc1,128)
                e4=decoder_block(e3,se4,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==3:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1)
                c2=vgg16.layers[3](c1)
                c2=vgg16.layers[4](c2)
                c2=vgg16.layers[5](c2)
                c3=vgg16.layers[6](c2)
                c3=vgg16.layers[7](c3)
                c3=vgg16.layers[8](c3)
                c3=vgg16.layers[9](c3)
                c4=vgg16.layers[10](c3)
                c4=vgg16.layers[11](c4)
                c4=vgg16.layers[12](c4)
                c4=vgg16.layers[13](c4)
                c5=vgg16.layers[14](c4)
                c5=vgg16.layers[15](c5)
                c5=vgg16.layers[16](c5)
                c5=vgg16.layers[17](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)
                se3=decoder_block(se2,c2,128)                
                #### Sub-Contracting Path
                sc1=vgg16.layers[6](se3)
                sc1=vgg16.layers[7](sc1)
                sc1=vgg16.layers[8](sc1)
                sc1=vgg16.layers[9](sc1)
                sc2=vgg16.layers[10](sc1)
                sc2=vgg16.layers[11](sc2)
                sc2=vgg16.layers[12](sc2)
                sc2=vgg16.layers[13](sc2)
                sc3=vgg16.layers[14](sc2)
                sc3=vgg16.layers[15](sc3)
                sc3=vgg16.layers[16](sc3)
                sc3=vgg16.layers[17](sc3)                
                #### Expanding Path
                e1=decoder_block(sc3,sc2,512)
                e2=decoder_block(e1,sc1,256)
                e3=decoder_block(e2,se3,128)
                e4=decoder_block(e3,c1,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==2:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1)
                c2=vgg16.layers[3](c1)
                c2=vgg16.layers[4](c2)
                c2=vgg16.layers[5](c2)
                c3=vgg16.layers[6](c2)
                c3=vgg16.layers[7](c3)
                c3=vgg16.layers[8](c3)
                c3=vgg16.layers[9](c3)
                c4=vgg16.layers[10](c3)
                c4=vgg16.layers[11](c4)
                c4=vgg16.layers[12](c4)
                c4=vgg16.layers[13](c4)
                c5=vgg16.layers[14](c4)
                c5=vgg16.layers[15](c5)
                c5=vgg16.layers[16](c5)
                c5=vgg16.layers[17](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)                                
                #### Sub-Contracting Path
                sc1=vgg16.layers[10](se2)
                sc1=vgg16.layers[11](sc1)
                sc1=vgg16.layers[12](sc1)
                sc1=vgg16.layers[13](sc1)
                sc2=vgg16.layers[14](sc1)
                sc2=vgg16.layers[15](sc2)
                sc2=vgg16.layers[16](sc2)
                sc2=vgg16.layers[17](sc2)                
                #### Expanding Path
                e1=decoder_block(sc2,sc1,512)
                e2=decoder_block(e1,se2,256)
                e3=decoder_block(e2,c2,128)
                e4=decoder_block(e3,c1,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==1:
                 #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1)
                c2=vgg16.layers[3](c1)
                c2=vgg16.layers[4](c2)
                c2=vgg16.layers[5](c2)
                c3=vgg16.layers[6](c2)
                c3=vgg16.layers[7](c3)
                c3=vgg16.layers[8](c3)
                c3=vgg16.layers[9](c3)
                c4=vgg16.layers[10](c3)
                c4=vgg16.layers[11](c4)
                c4=vgg16.layers[12](c4)
                c4=vgg16.layers[13](c4)
                c5=vgg16.layers[14](c4)
                c5=vgg16.layers[15](c5)
                c5=vgg16.layers[16](c5)
                c5=vgg16.layers[17](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)                
                #### Sub-Contracting Path
                sc1=vgg16.layers[14](se1)
                sc1=vgg16.layers[15](sc1)
                sc1=vgg16.layers[16](sc1)
                sc1=vgg16.layers[17](sc1)                
                #### Expanding Path
                e1=decoder_block(sc1,se1,512)
                e2=decoder_block(e1,c3,256)
                e3=decoder_block(e2,c2,128)
                e4=decoder_block(e3,c1,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==3 and dd==3:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1)
                c2=vgg16.layers[3](c1)
                c2=vgg16.layers[4](c2)
                c2=vgg16.layers[5](c2)
                c3=vgg16.layers[6](c2)
                c3=vgg16.layers[7](c3)
                c3=vgg16.layers[8](c3)
                c3=vgg16.layers[9](c3)
                c4=vgg16.layers[10](c3)
                c4=vgg16.layers[11](c4)
                c4=vgg16.layers[12](c4)
                c4=vgg16.layers[13](c4)
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)
                se2=decoder_block(se1,c2,128)
                se3=decoder_block(se2,c1,64)                
                #### Sub-Contracting Path
                sc1=vgg16.layers[3](se3)
                sc1=vgg16.layers[4](sc1)
                sc1=vgg16.layers[5](sc1)
                sc2=vgg16.layers[6](sc1)
                sc2=vgg16.layers[7](sc2)
                sc2=vgg16.layers[8](sc2)
                sc2=vgg16.layers[9](sc2)
                sc3=vgg16.layers[10](sc2)
                sc3=vgg16.layers[11](sc3)
                sc3=vgg16.layers[12](sc3)
                sc3=vgg16.layers[13](sc3)                
                #### Expanding Path
                e1=decoder_block(sc3,sc2,256)
                e2=decoder_block(e1,sc1,128)
                e3=decoder_block(e2,se3,64)
                dropout = Dropout(0.8)(e3)
            elif KK==3 and dd==2:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1) # 256x512x64
                c2=vgg16.layers[3](c1) # 128x256x64
                c2=vgg16.layers[4](c2) # 128x256x128
                c2=vgg16.layers[5](c2) # 128x256x128
                c3=vgg16.layers[6](c2) # 64x128x128
                c3=vgg16.layers[7](c3) # 64x128x256
                c3=vgg16.layers[8](c3) # 64x128x256
                c3=vgg16.layers[9](c3) # 64x128x256
                c4=vgg16.layers[10](c3)# 32x64x256
                c4=vgg16.layers[11](c4)# 32x64x512
                c4=vgg16.layers[12](c4)# 32x64x512
                c4=vgg16.layers[13](c4)# 32x64x512
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)  # 64x128x256
                se2=decoder_block(se1,c2,128) # 128x256x128                  
                #### Sub-Contracting Path
                sc1=vgg16.layers[6](se2) # 64x128x128
                sc1=vgg16.layers[7](sc1) # 64x128x256
                sc1=vgg16.layers[8](sc1) # 64x128x256
                sc1=vgg16.layers[9](sc1) # 64x128x256
                sc2=vgg16.layers[10](sc1)# 32x64x256
                sc2=vgg16.layers[11](sc2)# 32x64x512
                sc2=vgg16.layers[12](sc2)# 32x64x512
                sc2=vgg16.layers[13](sc2)# 32x64x512
                #### Expanding Path
                e1=decoder_block(sc2,sc1,256)
                e2=decoder_block(e1,se2,128)
                e3=decoder_block(e2,c1,64)
                dropout = Dropout(0.8)(e3)
            elif KK==3 and dd==1:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1) # 256x512x64
                c2=vgg16.layers[3](c1) # 128x256x64
                c2=vgg16.layers[4](c2) # 128x256x128
                c2=vgg16.layers[5](c2) # 128x256x128
                c3=vgg16.layers[6](c2) # 64x128x128
                c3=vgg16.layers[7](c3) # 64x128x256
                c3=vgg16.layers[8](c3) # 64x128x256
                c3=vgg16.layers[9](c3) # 64x128x256
                c4=vgg16.layers[10](c3)# 32x64x256
                c4=vgg16.layers[11](c4)# 32x64x512
                c4=vgg16.layers[12](c4)# 32x64x512
                c4=vgg16.layers[13](c4)# 32x64x512
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)  # 64x128x256
                #### Sub-Contracting Path
                sc1=vgg16.layers[10](se1)# 32x64x256
                sc1=vgg16.layers[11](sc1)# 32x64x512
                sc1=vgg16.layers[12](sc1)# 32x64x512
                sc1=vgg16.layers[13](sc1)# 32x64x512
                #### Expanding Path
                e1=decoder_block(sc1,se1,256)
                e2=decoder_block(e1,c2,128)
                e3=decoder_block(e2,c1,64)
                dropout = Dropout(0.8)(e3)
            elif KK==2 and dd==2:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1) # 256x512x64
                c2=vgg16.layers[3](c1) # 128x256x64
                c2=vgg16.layers[4](c2) # 128x256x128
                c2=vgg16.layers[5](c2) # 128x256x128
                c3=vgg16.layers[6](c2) # 64x128x128
                c3=vgg16.layers[7](c3) # 64x128x256
                c3=vgg16.layers[8](c3) # 64x128x256
                c3=vgg16.layers[9](c3) # 64x128x256
                #### Sub-Expanding Path
                se1=decoder_block(c3,c2,128)  # 128x256x128
                se2=decoder_block(se1,c1,64) # 256x512x64
                #### Sub-Contracting Path
                sc1=vgg16.layers[3](se2) # 128x256x64
                sc1=vgg16.layers[4](sc1) # 128x256x128
                sc1=vgg16.layers[5](sc1) # 128x256x128
                sc2=vgg16.layers[6](sc1) # 64x128x128
                sc2=vgg16.layers[7](sc2) # 64x128x256
                sc2=vgg16.layers[8](sc2) # 64x128x256
                sc2=vgg16.layers[9](sc2) # 64x128x256
                #### Expanding Path
                e1=decoder_block(sc2,sc1,128) # 
                e2=decoder_block(e1,se2,64)
                dropout = Dropout(0.8)(e2)
            elif KK==2 and dd==1:
                #### Contracting Path
                c1=vgg16.layers[1](xx)
                c1=vgg16.layers[2](c1) # 256x512x64
                c2=vgg16.layers[3](c1) # 128x256x64
                c2=vgg16.layers[4](c2) # 128x256x128
                c2=vgg16.layers[5](c2) # 128x256x128
                c3=vgg16.layers[6](c2) # 64x128x128
                c3=vgg16.layers[7](c3) # 64x128x256
                c3=vgg16.layers[8](c3) # 64x128x256
                c3=vgg16.layers[9](c3) # 64x128x256
                #### Sub-Expanding Path
                se1=decoder_block(c3,c2,128)  # 128x256x128
                #### Sub-Contracting Path                
                sc1=vgg16.layers[6](se1) # 64x128x128
                sc1=vgg16.layers[7](sc1) # 64x128x256
                sc1=vgg16.layers[8](sc1) # 64x128x256
                sc1=vgg16.layers[9](sc1) # 64x128x256
                #### Expanding Path
                e1=decoder_block(sc1,se1,128) # 128x256x128
                e2=decoder_block(e1,c1,64) # 256x512x64
                dropout = Dropout(0.8)(e2)                
            """ Output """
            outputs = Conv2D(1, 1, padding="same", activation=Final_Act)(dropout)
            model = Model(inputs, outputs, name='VGG16_EUNet{K}{d}'.format(K=KK,d=dd))
        elif backbone =='vgg19':
            inputs = Input(input_size)
            vgg19 = keras.applications.VGG19(include_top=False, 
                                             weights="imagenet", 
                                             input_tensor=inputs)
            xx = Lambda(lambda x: x/255)(inputs)
            if KK==4 and dd==4:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                c5=vgg19.layers[16](c4)
                c5=vgg19.layers[17](c5)
                c5=vgg19.layers[18](c5)
                c5=vgg19.layers[19](c5)
                c5=vgg19.layers[20](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)
                se3=decoder_block(se2,c2,128)
                se4=decoder_block(se3,c1,64)
                #### Sub-Contracting Path
                sc1=vgg19.layers[3](se4)
                sc1=vgg19.layers[4](sc1)
                sc1=vgg19.layers[5](sc1)
                sc2=vgg19.layers[6](sc1) # maxpooling
                sc2=vgg19.layers[7](sc2)
                sc2=vgg19.layers[8](sc2)
                sc2=vgg19.layers[9](sc2)
                sc2=vgg19.layers[10](sc2)
                sc3=vgg19.layers[11](sc2) # maxpooling
                sc3=vgg19.layers[12](sc3)
                sc3=vgg19.layers[13](sc3)
                sc3=vgg19.layers[14](sc3)
                sc3=vgg19.layers[15](sc3)
                sc4=vgg19.layers[16](sc3) # maxpooling
                sc4=vgg19.layers[17](sc4)
                sc4=vgg19.layers[18](sc4)
                sc4=vgg19.layers[19](sc4)
                sc4=vgg19.layers[20](sc4)
                #### Expanding Path
                e1=decoder_block(sc4,sc3,512)
                e2=decoder_block(e1,sc2,256)
                e3=decoder_block(e2,sc1,128)
                e4=decoder_block(e3,se4,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==3:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                c5=vgg19.layers[16](c4)
                c5=vgg19.layers[17](c5)
                c5=vgg19.layers[18](c5)
                c5=vgg19.layers[19](c5)
                c5=vgg19.layers[20](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)
                se3=decoder_block(se2,c2,128)                
                #### Sub-Contracting Path
                sc1=vgg19.layers[6](se3)
                sc1=vgg19.layers[7](sc1)
                sc1=vgg19.layers[8](sc1)
                sc1=vgg19.layers[9](sc1)
                sc1=vgg19.layers[10](sc1)
                sc2=vgg19.layers[11](sc1)
                sc2=vgg19.layers[12](sc2)
                sc2=vgg19.layers[13](sc2)
                sc2=vgg19.layers[14](sc2)
                sc2=vgg19.layers[15](sc2)
                sc3=vgg19.layers[16](sc2)
                sc3=vgg19.layers[17](sc3)                
                sc3=vgg19.layers[18](sc3)
                sc3=vgg19.layers[19](sc3)
                sc3=vgg19.layers[20](sc3)
                #### Expanding Path
                e1=decoder_block(sc3,sc2,512)
                e2=decoder_block(e1,sc1,256)
                e3=decoder_block(e2,se3,128)
                e4=decoder_block(e3,c1,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==2:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                c5=vgg19.layers[16](c4)
                c5=vgg19.layers[17](c5)
                c5=vgg19.layers[18](c5)
                c5=vgg19.layers[19](c5)
                c5=vgg19.layers[20](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)
                se2=decoder_block(se1,c3,256)                                
                #### Sub-Contracting Path
                sc1=vgg19.layers[11](se2)
                sc1=vgg19.layers[12](sc1)
                sc1=vgg19.layers[13](sc1)
                sc1=vgg19.layers[14](sc1)
                sc1=vgg19.layers[15](sc1)                
                sc2=vgg19.layers[16](sc1)
                sc2=vgg19.layers[17](sc2)                
                sc2=vgg19.layers[18](sc2)
                sc2=vgg19.layers[19](sc2)
                sc2=vgg19.layers[20](sc2)
                #### Expanding Path
                e1=decoder_block(sc2,sc1,512)
                e2=decoder_block(e1,se2,256)
                e3=decoder_block(e2,c2,128)
                e4=decoder_block(e3,c1,64)               
                dropout = Dropout(0.8)(e4)
            elif KK==4 and dd==1:
                 #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                c5=vgg19.layers[16](c4)
                c5=vgg19.layers[17](c5)
                c5=vgg19.layers[18](c5)
                c5=vgg19.layers[19](c5)
                c5=vgg19.layers[20](c5)
                #### Sub-Expanding Path
                se1=decoder_block(c5,c4,512)                
                #### Sub-Contracting Path
                sc1=vgg19.layers[16](se1)
                sc1=vgg19.layers[17](sc1)
                sc1=vgg19.layers[18](sc1)
                sc1=vgg19.layers[19](sc1)                
                sc1=vgg19.layers[20](sc1)
                #### Expanding Path
                e1=decoder_block(sc1,se1,512)
                e2=decoder_block(e1,c3,256)
                e3=decoder_block(e2,c2,128)
                e4=decoder_block(e3,c1,64)
                dropout = Dropout(0.8)(e4)
            elif KK==3 and dd==3:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)
                se2=decoder_block(se1,c2,128)
                se3=decoder_block(se2,c1,64)                
                #### Sub-Contracting Path
                sc1=vgg19.layers[3](se3)
                sc1=vgg19.layers[4](sc1)
                sc1=vgg19.layers[5](sc1)
                sc2=vgg19.layers[6](sc1) # maxpooling
                sc2=vgg19.layers[7](sc2)
                sc2=vgg19.layers[8](sc2)
                sc2=vgg19.layers[9](sc2)
                sc2=vgg19.layers[10](sc2)
                sc3=vgg19.layers[11](sc2) # maxpooling
                sc3=vgg19.layers[12](sc3)
                sc3=vgg19.layers[13](sc3)
                sc3=vgg19.layers[14](sc3)
                sc3=vgg19.layers[15](sc3)               
                #### Expanding Path
                e1=decoder_block(sc3,sc2,256)
                e2=decoder_block(e1,sc1,128)
                e3=decoder_block(e2,se3,64)
                dropout = Dropout(0.8)(e3)
            elif KK==3 and dd==2:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)  # 64x128x256
                se2=decoder_block(se1,c2,128) # 128x256x128                  
                #### Sub-Contracting Path
                sc1=vgg19.layers[6](se2) # 64x128x128
                sc1=vgg19.layers[7](sc1) # 64x128x256
                sc1=vgg19.layers[8](sc1) # 64x128x256
                sc1=vgg19.layers[9](sc1) # 64x128x256
                sc1=vgg19.layers[10](sc1)# 32x64x256
                sc2=vgg19.layers[11](sc1)# 32x64x512
                sc2=vgg19.layers[12](sc2)# 32x64x512
                sc2=vgg19.layers[13](sc2)# 32x64x512
                sc2=vgg19.layers[14](sc2)
                sc2=vgg19.layers[15](sc2)                
                #### Expanding Path
                e1=decoder_block(sc2,sc1,256)
                e2=decoder_block(e1,se2,128)
                e3=decoder_block(e2,c1,64)
                dropout = Dropout(0.8)(e3)
            elif KK==3 and dd==1:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                c4=vgg19.layers[11](c3)
                c4=vgg19.layers[12](c4)
                c4=vgg19.layers[13](c4)
                c4=vgg19.layers[14](c4)
                c4=vgg19.layers[15](c4)
                #### Sub-Expanding Path
                se1=decoder_block(c4,c3,256)  # 64x128x256
                #### Sub-Contracting Path
                sc1=vgg19.layers[11](se1)# 32x64x256
                sc1=vgg19.layers[12](sc1)# 32x64x512
                sc1=vgg19.layers[13](sc1)# 32x64x512
                sc1=vgg19.layers[14](sc1)# 32x64x512
                sc1=vgg19.layers[15](sc1)# 32x64x512
                #### Expanding Path
                e1=decoder_block(sc1,se1,256)
                e2=decoder_block(e1,c2,128)
                e3=decoder_block(e2,c1,64)
                dropout = Dropout(0.8)(e3)
            elif KK==2 and dd==2:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                #### Sub-Expanding Path
                se1=decoder_block(c3,c2,128)  # 128x256x128
                se2=decoder_block(se1,c1,64) # 256x512x64
                #### Sub-Contracting Path
                sc1=vgg19.layers[3](se2) # 128x256x64
                sc1=vgg19.layers[4](sc1) # 128x256x128
                sc1=vgg19.layers[5](sc1) # 128x256x128
                sc2=vgg19.layers[6](sc1) # 64x128x128
                sc2=vgg19.layers[7](sc2) # 64x128x256
                sc2=vgg19.layers[8](sc2) # 64x128x256
                sc2=vgg19.layers[9](sc2) # 64x128x256
                sc2=vgg19.layers[10](sc2) # 64x128x256
                #### Expanding Path
                e1=decoder_block(sc2,sc1,128) # 
                e2=decoder_block(e1,se2,64)
                dropout = Dropout(0.8)(e2)
            elif KK==2 and dd==1:
                #### Contracting Path
                c1=vgg19.layers[1](xx)
                c1=vgg19.layers[2](c1)
                c2=vgg19.layers[3](c1)
                c2=vgg19.layers[4](c2)
                c2=vgg19.layers[5](c2)
                c3=vgg19.layers[6](c2)
                c3=vgg19.layers[7](c3)
                c3=vgg19.layers[8](c3)
                c3=vgg19.layers[9](c3)
                c3=vgg19.layers[10](c3)
                #### Sub-Expanding Path
                se1=decoder_block(c3,c2,128)  # 128x256x128
                #### Sub-Contracting Path                
                sc1=vgg19.layers[6](se1) # 64x128x128
                sc1=vgg19.layers[7](sc1) # 64x128x256
                sc1=vgg19.layers[8](sc1) # 64x128x256
                sc1=vgg19.layers[9](sc1) # 64x128x256
                sc1=vgg19.layers[10](sc1)
                #### Expanding Path
                e1=decoder_block(sc1,se1,128) # 128x256x128
                e2=decoder_block(e1,c1,64) # 256x512x64
                dropout = Dropout(0.8)(e2)                
            """ Output """
            outputs = Conv2D(1, 1, padding="same", activation=Final_Act)(dropout)
            model = Model(inputs, outputs, name='VGG19_EUNet{K}{d}'.format(K=KK,d=dd))
    return model
