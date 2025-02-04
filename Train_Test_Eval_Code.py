# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:00:37 2022

@author: Ashkan
"""

import os
from glob import glob
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize


from tensorflow import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from FlexUNET_Model() import FlexUNET_Model

### Preprocessing Stage based on Color Transformation and Patch extraction ###
'''
global Preproc
Preproc = 'n'  # 'y' or 'n' as transformation from rgb to Lab

IMG_HEIGHT = np.int(np.fix(IMG_HEIGHT / (2 ** KK)) * (2 ** KK))
IMG_WIDTH = np.int(np.fix(IMG_WIDTH / (2 ** KK)) * (2 ** KK))

# 4. Define Pre_Processing Function (Region of Interest Extraction _ ROI)

Train_Mask_List = sorted(next(os.walk(TRAIN_MASK_PATH))[2])
Test_Mask_List = sorted(next(os.walk(TEST_MASK_PATH))[2])

NL = 9 + (KK + dd - 1) * 5  # No. of layers for EW-Net architecture
MD = [None] * (NL)  # EW_Net Layers combination

NoPatch = 1  # it must be a quadrate integer, preferablely a power of 4
SqNP = np.int(NoPatch ** .5)
h = np.int((IMG_HEIGHT / (KK ** 2)) * (KK ** 2) / SqNP)
w = np.int((IMG_WIDTH / (KK ** 2)) * (KK ** 2) / SqNP)


def Data_Train():
    if Preproc == 'y':
        Train_X = np.zeros((len(Train_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                            np.int(IMG_WIDTH / SqNP), IMG_CHANNELS), dtype=np.float32)
    else:
        Train_X = np.zeros((len(Train_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                            np.int(IMG_WIDTH / SqNP), IMG_CHANNELS), dtype=np.uint8)

    Train_Y = np.zeros((len(Train_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                        np.int(IMG_WIDTH / SqNP), 1), dtype=np.bool)

    n = 0
    
    if PATH.find('CCE')!=-1:
        TEXT = '{}/*.png'
        TEXT1 = '{}/{}.jpg'
    else:
        TEXT = '{}/*.tif'
        TEXT1 = '{}/{}.tif'

    for mask_path in glob.glob(TEXT.format(TRAIN_MASK_PATH)):

        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = TEXT1.format(TRAIN_IMAGE_PATH, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        # Temporary image variable
        TempI = resize(image[:, :, :IMG_CHANNELS],
                       (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                       mode='constant',
                       anti_aliasing=True,
                       preserve_range=True)

        if Preproc == 'y':
            TempI = rgb2lab(TempI / 255)  # rgb to Lab

        # Temporary mask variable
        TempM = np.expand_dims(resize(mask,
                                      (IMG_HEIGHT, IMG_WIDTH),
                                      mode='constant',
                                      anti_aliasing=True,
                                      preserve_range=True), axis=-1)

        for cc1 in np.arange(SqNP):
            for cc2 in np.arange(SqNP):
                Train_X[n + SqNP * cc1 + cc2] = TempI[cc1 * h:(cc1 + 1) * h, \
                                                cc2 * w:(cc2 + 1) * w]
                Train_Y[n + SqNP * cc1 + cc2] = TempM[cc1 * h:(cc1 + 1) * h, \
                                                cc2 * w:(cc2 + 1) * w]

        n += NoPatch

    return Train_X, Train_Y


Train_Inputs, Train_Masks = Data_Train()

if Train_augmentation == 'y':
    seq1 = iaa.Rot90(1)
    images_aug1, segmaps_aug1 = seq1(images=Train_Inputs, \
                                     segmentation_maps=Train_Masks)
    seq2 = iaa.Fliplr(1)
    images_aug2, segmaps_aug2 = seq2(images=Train_Inputs, \
                                     segmentation_maps=Train_Masks)
    seq3 = iaa.Flipud(1)
    images_aug3, segmaps_aug3 = seq3(images=Train_Inputs, \
                                     segmentation_maps=Train_Masks)
    seq4 = iaa.Rot90(3)
    images_aug4, segmaps_aug4 = seq4(images=Train_Inputs, \
                                     segmentation_maps=Train_Masks)
    Train_Inputs = np.concatenate((images_aug1, images_aug2, images_aug3, images_aug4))
    Train_Masks = np.concatenate((segmaps_aug1, segmaps_aug2, segmaps_aug3, segmaps_aug4))


def Data_Test():
    if Preproc == 'y':
        Test_X = np.zeros((len(Test_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                           np.int(IMG_WIDTH / SqNP), IMG_CHANNELS), dtype=np.float32)
    else:
        Test_X = np.zeros((len(Test_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                           np.int(IMG_WIDTH / SqNP), IMG_CHANNELS), dtype=np.uint8)

    Test_Y = np.zeros((len(Test_Mask_List) * NoPatch, np.int(IMG_HEIGHT / SqNP), \
                       np.int(IMG_WIDTH / SqNP), 1), dtype=np.bool)

    n = 0
    if PATH.find('CCE')!=-1:
        TEXT = '{}/*.png'
        TEXT1 = '{}/{}.jpg'
    else:
        TEXT = '{}/*.tif'
        TEXT1 = '{}/{}.tif'
        
    for mask_path in glob.glob(TEXT.format(TEST_MASK_PATH)):

        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = TEXT1.format(TEST_IMAGE_PATH, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        # Temporary image variable
        TempI = resize(image[:, :, :IMG_CHANNELS],
                       (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                       mode='constant',
                       anti_aliasing=True,
                       preserve_range=True)

        if Preproc == 'y':
            TempI = rgb2lab(TempI / 255)  # rgb to Lab

        # Temporary mask variable    
        TempM = np.expand_dims(resize(mask,
                                      (IMG_HEIGHT, IMG_WIDTH),
                                      mode='constant',
                                      anti_aliasing=True,
                                      preserve_range=True), axis=-1)

        for cc1 in np.arange(SqNP):
            for cc2 in np.arange(SqNP):
                Test_X[n + SqNP * cc1 + cc2] = TempI[cc1 * h:(cc1 + 1) * h, \
                                               cc2 * w:(cc2 + 1) * w]
                Test_Y[n + SqNP * cc1 + cc2] = TempM[cc1 * h:(cc1 + 1) * h, \
                                               cc2 * w:(cc2 + 1) * w]

        n += NoPatch

    return Test_X, Test_Y


Test_Inputs, Test_Masks = Data_Test()

if Test_augmentation == 'y':
    seq1 = iaa.Rot90(1)
    images_aug1, segmaps_aug1 = seq1(images=Test_Inputs, \
                                     segmentation_maps=Test_Masks)
    seq2 = iaa.Fliplr(1)
    images_aug2, segmaps_aug2 = seq2(images=Test_Inputs, \
                                     segmentation_maps=Test_Masks)
    seq3 = iaa.Flipud(1)
    images_aug3, segmaps_aug3 = seq3(images=Test_Inputs, \
                                     segmentation_maps=Test_Masks)
    seq4 = iaa.Rot90(3)
    images_aug4, segmaps_aug4 = seq4(images=Test_Inputs, \
                                     segmentation_maps=Test_Masks)
    Test_Inputs = np.concatenate((images_aug1, images_aug2, images_aug3, images_aug4))
    Test_Masks = np.concatenate((segmaps_aug1, segmaps_aug2, segmaps_aug3, segmaps_aug4))

## 4.1. Show The Results in Preprocessing Stage

# IndexN=np.uint16(np.random.uniform(0, len(Train_Mask_List)))
IndexN = random.randint(0, len(Train_Inputs) - 1)

RR = IndexN % NoPatch
if Preproc == 'y':
    TempN = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float)
else:
    TempN = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

TempP = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for ii in np.arange(SqNP):
    for jj in np.arange(SqNP):
        PT = ii * SqNP + jj
        TempN[ii * h:(ii + 1) * h, jj * w:(jj + 1) * w, :IMG_CHANNELS] = \
            Train_Inputs[IndexN + (PT - RR)]
        TempP[ii * h:(ii + 1) * h, jj * w:(jj + 1) * w] = \
            Train_Masks[IndexN + (PT - RR)]

if Preproc == 'y':
    TempN = lab2rgb(TempN)

print('Region_of_Interest_Image')
imshow(TempN)
plt.show()

print('Region_of_Interest_Mask')
imshow(np.squeeze(TempP))
plt.show()
'''
##############################################################################

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [256, 512, 3]


train_files = []
train_files=glob('Original\\*.png') # or any other formats like *.jpg
mask_files = glob('GT\\*.png') # or any other formats like *.bmp or *.jpg


df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)



inputs_size = input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(IMG_CHANNELS, IMG_WIDTH),
                    seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="filename",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return abs(dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred))


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
############################################################################
# Adjustment of training Parameters
epochs = 20
batchSIZE = 2
Patience = 7
learning_rate = 1e-5
############################################################################
train_generator_args = dict(rotation_range=0.5,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')

decay_rate = learning_rate / epochs


opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, 
                            beta_2=0.999, epsilon=None, decay=decay_rate,
                            amsgrad=False)

train_gen = train_generator(df_train, batchSIZE, train_generator_args,
                            target_size=(IMG_HEIGHT, IMG_WIDTH))

x_img, x_mask= next(train_gen)


valid_gen = train_generator(df_val, batchSIZE,
                                  dict(),
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))
###############################################################################
# Hyperparameter adjustment 
##### List of popular activation functions ####################################
# Sigmoid (Logistic) --> 'sigmoid'
# Hyperbolic Tangent (Tanh) --> 'tanh'
# Rectified Linear Unit (ReLU) --> 'relu'
# Leaky ReLU --> from tensorflow.keras.layers import LeakyReLU --> activation = LeakyReLU(alpha=0.01)
# Parametric Leaky ReLU (PReLU) --> from tensorflow.keras.layers import PReLU --> activation = PReLU()
# Exponential Linear Units (ELU) --> 'elu' or from tensorflow.keras.activations import elu --> 
# Scaled Exponential Linear Unit (SELU) --> 'selu' & kernel_initializer='lecun_normal'
Final_Act = 'sigmoid'#'selu'
###############################################################################
# Network selection
NN, KK, dd = 5, 4, 1 # The KK & dd hyperparameters can get different values but ...
                     # for especial Backbones like 'vgg16' and 'vgg19', their values
                     # should be as follows:
                     # for vgg16 and vgg19: KK, dd = [4,4] or [4,3] or [4,2] or [4,1] or
                     # [3,3] or [3,2] or [3,1] or [2,2] or [2,1]
Backbone='vgg19' # 'None' or 'vgg16' or 'vgg19'
model=FlexUNET_Model.Flex_U_Net_Segmentation(NN, KK, dd, Final_Act, backbone = Backbone)
###############################################################################
model.summary()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
import keras.utils.vis_utils
from importlib import reload
reload(keras.utils.vis_utils)
from tensorflow.keras.utils import plot_model

model.compile(loss=bce_dice_loss, optimizer=opt,
              metrics=['binary_accuracy', dice_coef, iou])


class loss_history(keras.callbacks.Callback):
    def __init__(self, x=0):
        self.x = x
    def on_epoch_begin(self, epoch, logs={}):
        if Final_Act == 'tanh':
            pred_thresh = 0.01
        else:
            pred_thresh = 0.5
        plt.imshow(x_img[self.x])
        plt.show()
        plt.imshow(np.squeeze(x_mask[self.x]))
        plt.show()
        preds_train_1 = self.model.predict(np.expand_dims(x_img[self.x], axis=0))
        plt.imshow(np.squeeze((preds_train_1[0] > pred_thresh).astype(float)))
        plt.show()

imageset = 'OC'# any type of images 'OC' or 'CCE', etc.
backbone = 'Flex_Unet_{NN}-{KK}-{dd}-{backbone}'.format(NN=NN, KK=KK, dd=dd, backbone=Backbone)#'RessUnet'#'VGG19_Unet'#'Flex_Unet'

No_show =0
if No_show!=1:
    plot_model(model, to_file='model_plot_{ImgForm}_{Network}.png'.\
               format(ImgForm=imageset, Network=backbone), 
               show_shapes=True, show_layer_names=True)


model_h5 = 'home/Python_Codes/saved_model/{backbone}-{imageset}_model' \
    .format(backbone=backbone, \
            imageset=imageset)
filepath = "home/Python_Codes/saved_model/weights-best.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
earlystopper = EarlyStopping(patience=Patience, verbose=1)
results = model.fit(train_gen,
                    steps_per_epoch=len(df_train) / batchSIZE,
                    epochs=epochs,
                    validation_data=valid_gen,
                    validation_steps=len(df_val) / batchSIZE, verbose=1,
                    callbacks=[ checkpointer, earlystopper,loss_history()])

model_json = model.to_json()
with open('{model_h5}.json'.format(model_h5=model_h5), "w") as json_file:
    json_file.write(model_json)
# b) serialize weights to HDF5
model.save_weights('{model_h5}.hdf5'.format(model_h5=model_h5))
print("Saved model to disk")
import pickle
from os import path

with open(path.join('home\\Python_Codes\\Results_hist.pkl'), "wb") as DD:
    pickle.dump(results.history,DD)
###############################################################################
Results_hist = results.history

plt.plot(Results_hist['loss'])
plt.plot(Results_hist['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(np.multiply(Results_hist['binary_accuracy'], 100))
plt.plot(np.multiply(Results_hist['val_binary_accuracy'], 100))
plt.title('Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()


###############################################################################
# Testing Phase

from skimage.io import imread#, imshow
test_Inputs = []
test_Masks = []
for ii in range(len(df_test)):
    temp = imread(df_test.iloc[ii]['filename'])
    temp = np.uint8(resize(temp,
                    (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    mode='constant',
                    anti_aliasing=True,
                    preserve_range=True))
    test_Inputs.append(temp)
    temp = imread(df_test.iloc[ii]['mask'])
    temp = np.expand_dims(resize(temp,
                                  (IMG_HEIGHT, IMG_WIDTH),
                                  mode='constant',
                                  anti_aliasing=True,
                                  preserve_range=True), axis=-1)
    test_Masks.append(temp>0)

test_Masks = np.array(1*test_Masks)
test_Inputs = np.array(test_Inputs)

# test_Inputs1 = test_Inputs
# test_Inputs = (lambda x: x/(255**2))(test_Inputs)
test_Inputs = (lambda x: x/(255))(test_Inputs)


from ttictoc import tic, toc
# tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
# with tf.device('/CPU:0'):
tic()
# preds_test = model.predict(test_gen, verbose=2, batch_size=200)
preds_test = model.predict(test_Inputs, verbose=2, batch_size=2)
TT=toc()
# print('Inference time for {} training Images = {}'.format((ii)*len(Train_Inputs),TT))
if No_show !=1:
    fig, axs =  plt.subplots(3)
    fig.suptitle('Sample Original Image + Mask + Prediction')
    axs[0].imshow(np.uint8((test_Inputs[0])*(255)))
    axs[1].imshow(test_Masks[0]*1)
    axs[2].imshow(preds_test[0]>.0001)
    plt.show()

############# AP and PR-Curve

# import sklearn.metrics
import sklearn.metrics
import matplotlib.pyplot
import matplotlib.patches

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        # y_pred = [True if score >= threshold else False for score in pred_scores]
        y_pred = [True if score >= threshold else False for score in pred_scores]
        # y_pred1 = np.array(y_pred)
        # y_pred1 = np.expand_dims(y_pred1,1)
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=True)
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='binary', pos_label=True)
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


SS1=test_Masks.shape
y_true = test_Masks.reshape(np.prod(SS1),1)
pred_scores = preds_test.reshape(np.prod(SS1),1)


thresholds = np.arange(start=0.09, stop=1.09, step=0.09)

precisions, recalls = precision_recall_curve(y_true=y_true, 
                                             pred_scores=pred_scores,
                                             thresholds=thresholds)

f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))

plt.plot(recalls, precisions, linewidth=4, color="blue", zorder=0, label=backbone)
matplotlib.pyplot.scatter(recalls[9], precisions[9], zorder=1, linewidth=6)

matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
plt.legend()
matplotlib.pyplot.show()



AP = np.sum((np.subtract(recalls[:-1], recalls[1:])) * precisions[:-1])
print(AP)

### Save Precision-Recall for specific configuration of th deployed network
with open('Pr_Re_{Net}_{Final_Act}.pkl'.format(Net=backbone,Final_Act=Final_Act), 'wb') as f:
    pickle.dump([precisions, recalls], f)
    

# 12. Confusion Matrix Accuracy Cal. for Test Images

def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
    """
    A = np.uint8(np.squeeze(groundtruth_list))  # np.uint8(np.squeeze(Test_Masks[iix]))
    B = np.squeeze(predicted_list)  # np.squeeze(preds_test_t[iix])

    tp = np.count_nonzero(A * B)
    fn = np.count_nonzero(A) - tp
    fp = np.count_nonzero(B) - tp
    tn = (A.shape[0] * A.shape[1]) - (tp + fp + fn)

    return tn, fp, fn, tp

# import random
iix = random.randint(0, len(test_Inputs) - 1)
#iix = 10

TN, FP, FN, TP = get_confusion_matrix_elements(test_Masks[iix], preds_test[iix]>0.5)

# 13. Pretty Confusion Matrix Drawing

# imports
from pandas import DataFrame
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


#
import seaborn as sn

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')

#'twilight_shifted'#'twilight'#'hsv'#'plasma'#'viridis'#'Greys'#'cividis'#'Greens' #'gist_rainbow' #'Oranges'  # 'oranges'
cmap = 'twilight_shifted'

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap=cmap, fmt='.2f', fz=20,
                                 lw=1.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='x'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, fontsize=15, \
                       weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=35, fontsize=15, \
                       weight='bold')

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    plt.figure(figsize=(10, 7))

    sn.set(font_scale=1.4)  # for label size

    # titles and legends
    ax.set_title('Confusion matrix', weight='bold', fontsize=30)
    ax.set_xlabel(xlbl, weight='bold', fontsize=25)
    ax.set_ylabel(ylbl, weight='bold', fontsize=25)
    plt.tight_layout()  # set layout slim
    plt.show()


#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap=cmap,
                                    fmt='.2f', fz=20, lw=1.5, cbar=False, fig_size=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        without a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    if not columns:
        # labels axis integer:
        # columns = range(1, len(np.unique(y_test))+1)
        # labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' % i for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    conf_matrix = confusion_matrix(y_test, predictions)
    fz = 20
    fig_size = [9, 9]
    show_null_values = 2
    dataframe = DataFrame(conf_matrix, index=columns, columns=columns)
    pretty_plot_confusion_matrix(dataframe, fz=fz, cmap=cmap, figsize=fig_size,
                                 show_null_values=show_null_values,
                                 pred_val_axis=pred_val_axis)


array = np.array([[TP, FN],
                  [FP, TN]])

df_cm = DataFrame(array, index=["TP", "FP"], columns=["FN", "TN"])
cmap = 'twilight'#'twilight_shifted'#'hsv'#'plasma'#'viridis'#'Greys'#'cividis'#'Greens' #'gist_rainbow' #'Oranges'  # 'oranges'
pretty_plot_confusion_matrix(df_cm, cmap=cmap)

# 14. Schematic of proposed 
import cv2


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    #    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)
    for label1, mask in masks.items():
        color = colors[label1]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


alpha = 0.5
confusion_matrix_colors = {
    'tp': (0, 255, 255),  # cyan
    'fp': (255, 0, 255),  # magenta
    'fn': (255, 255, 0),  # yellow
    'tn': (0, 0, 0)  # black
}
A = np.uint8(np.squeeze(test_Masks[iix]))
B = (np.squeeze(preds_test[iix]>0.5))
validation_mask = get_confusion_matrix_overlaid_mask(test_Inputs[iix], A,
                                                     B, alpha, 
                                                     confusion_matrix_colors)
print('Cyan - TP')
print('Magenta - FP')
print('Yellow - FN')
print('Black - TN')
plt.imshow(validation_mask)
plt.axis('off')
plt.title('confusion matrix overlay mask')

# 15. Confusion Matrix for total of Test_Inputs

TP = 0
TN = 0
FN = 0
FP = 0

for t in np.arange(0, len(test_Inputs)):
    TN1, FP1, FN1, TP1 = get_confusion_matrix_elements(test_Masks[t], preds_test[t]>0.5)
    TP += TP1
    FP += FP1
    TN += TN1
    FN += FN1

array = np.array([[TP, FN],
                  [FP, TN]])

df_cm = pd.DataFrame(array, index=["TP", "FP"], columns=["FN", "TN"])

# df_cm = df_cm.style.set_properties({'font-size': '20pt'})
pretty_plot_confusion_matrix(df_cm, cmap=cmap)
plt.figure(figsize=(10, 7))

sn.set(font_scale=2.4)  # for label size

F_score = np.round(10000 * (2 * TP) / (2 * TP + FP + FN)) / 100
Accuracy = np.round(10000 * (TP + TN) / (TP + FP + FN + TN)) / 100
print('TP=', TP)
print('FP=', FP)
print('FN=', FN)
print('TN=', TN)
print('Accuracy= ', Accuracy)
print('F_Score=', F_score)


