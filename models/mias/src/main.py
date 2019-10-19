import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle
import random
import json
import minio
#tf.compat.v1.disable_eager_execution()
import io
import argparse
from minio.error import ResponseError
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--run-id', type=str, help='')
args = parser.parse_args()
run_id = args.run_id

L2_WEIGHT_DECAY = 1e-4

L_PATH = 'experts_anotation'
I_PATH = 'images'

print(L_PATH)
print(I_PATH)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import os

os.system('nvidia-smi')

minio_client = minio.Minio('minio:9000', access_key='admin', secret_key='password', secure=False)

from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient()

configs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

mlflow_client.download_artifacts(run_id, 'configs', dst_path=os.path.dirname(os.path.abspath(__file__)))

print(os.listdir('configs'))

train = None
with open(os.path.join(configs_path, 'train.json'), 'r') as f:
    train = json.load(f)

if train is None:
    exit(-1)

test = None
with open(os.path.join(configs_path, 'test.json'), 'r') as f:
    test = json.load(f)

if train is None:
    exit(-1)

mode = None
with open(os.path.join(configs_path, 'model.json'), 'r') as f:
    mode = json.load(f)

if train is None:
    exit(-1)
    
def anot_to_img(expert, it):
    expert = str(expert)
    it = "{0:0=3d}".format(it)
    contours = []
    try:
        data = minio_client.get_object('drions', os.path.join(L_PATH,'anotExpert'+expert+'_'+it+'.txt'))
        bio = io.BytesIO()
        for d in data.stream(32*1024):
            bio.write(d)
        bio.seek(0)
    except ResponseError as err:
        print(err)
    line = bio.readline().decode("utf-8") 
    while line:
        x = re.findall("\d+\.\d+", line)
        contours.append([int(float(x[0])), int(float(x[1]))])
        line = bio.readline().decode("utf-8") 
    bio.close()
    
    contours = np.array(contours)
    try:
        data = minio_client.get_object('drions', os.path.join(I_PATH,'image_'+it+'.jpg'))
        bio = io.BytesIO()
        for d in data.stream(32*1024):
            bio.write(d)
        bio.seek(0)
    except ResponseError as err:
        print(err)
    file_bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    bio.close()
    mask = np.zeros(img.shape, np.uint8)
    
    
    
    img = img[:,100:500]

    cv2.fillPoly(mask, pts =[contours], color=255)
    mask = mask[:,100:500]

    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_AREA)
    return img, mask

def show_mask(expert, it):
    img, mask = anot_to_img(expert, it)
    output = img.copy()
    alpha = 0.3
    cv2.addWeighted(mask, alpha, output, 1 - alpha, 0, output)

    plt.imshow(output)

show_mask(2,19)

img, mask = anot_to_img(2,19)

plt.imshow(img)
print(img.shape)

plt.imshow(mask)
print(mask.shape)

import random

def image_generator(files, batch_size = 32):
    while True: 
        #extract a random batch 
        batch = random.sample(files, batch_size)    

        #variables for collecting batches of inputs and outputs 
        batch_x = []
        batch_y = []

        for exp, it in batch:
            #get the masks. Note that masks are png files
            img, mask = anot_to_img(exp, it)

            if random.random() > 0.5:
                img = np.flip(img, axis=0)
                mask = np.flip(mask, axis=0)

            if random.random() > 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)

            batch_y.append(mask)
            batch_x.append(img)

        #preprocess a batch of images and masks 
        batch_x = np.array(batch_x)/255.
        batch_y = np.array(batch_y)/255.

        # opencv grayscale is (h,w) and we need (h,w,1)
        batch_x = np.expand_dims(batch_x,3)
        batch_y = np.expand_dims(batch_y,3)

        yield (batch_x, batch_y)      

def to_files(exp_range, it_range):
    files = []
    for it in it_range:
        for exp in exp_range:
            files.append((exp, it))
    return files

train_batch_size = train['batch_size']
test_batch_size = test['batch_size']

train_files = to_files(range(1,3), range(1,101))
test_files  = to_files(range(1,3), range(101,111))

train_generator = image_generator(train_files, batch_size = train_batch_size)
test_generator  = image_generator(test_files, batch_size = test_batch_size)

x, y = next(train_generator)

plt.axis('off')
img = x[0].squeeze()
img = np.stack((img,)*3, axis=-1)
msk = y[0].squeeze()
msk = np.stack((msk,)*3, axis=-1)

plt.imshow( np.concatenate([img, msk, img*msk], axis = 1))

def unet(sz = (256, 256, 1)):
    x = Input(sz)
    inputs = x

    #down sampling 
    f = 8
    layers = []

    for i in range(0, mode['depth']+1):
        x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
        x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
        layers.append(x)
        x = MaxPooling2D() (x)
        f = f*2
    ff2 = 64 

    #bottleneck 
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
    x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 

    #upsampling 
    for i in range(0, mode['depth']):
        ff2 = ff2//2
        f = f // 2 
        x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
        x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j -1 


    #classification 
    x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
    x = Conv2D(f, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)
    outputs = Conv2D(1, 1, activation='sigmoid', use_bias=False, kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)) (x)

    #model creation 
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

model = unet()
model.summary()

loss_object = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=train['lr'])

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='test_iou')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_iou(labels, tf.math.round(predictions))
    return predictions

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_iou(labels, tf.math.round(predictions))
    return predictions

from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient()

EPOCHS = train['epochs']
TRAIN_STEPS = train['steps_per_epoch']
TEST_STEPTS = test['steps_per_epoch']
st = 0
last_test_loss = 0
last_st = 0
for epoch in range(EPOCHS):
    for i in range(TRAIN_STEPS):
        images, labels = next(train_generator)
        train_step(images, labels)
    for i in range(TEST_STEPTS):
        test_images, test_labels = next(test_generator)
        predictions = test_step(test_images, test_labels)
    
    crr_test_loss = test_loss.result().numpy()
    
    if st > train['early_stop_min_epochs']:
        if crr_test_loss < last_test_loss and (last_test_loss - crr_test_loss) > train['early_stop_delta']:
            last_st = st
    else:
        last_st = st
    last_test_loss = test_loss.result().numpy()
    
    st+=1
    
    if (st - last_st) > train['early_stop_max_epochs']:
        break
    
    model.save('model.h5')
    mlflow_client.log_artifact(run_id, 'model.h5')

    pd1 = predictions.numpy()[0]
    im1 = test_images[0]
    lb1 = test_labels[0]
    
    cv2.imwrite('pred.jpg', (pd1[:,:,0]*255).astype(np.uint8))
    cv2.imwrite('imag.jpg', (im1[:,:,0]*255).astype(np.uint8))
    cv2.imwrite('lbl.jpg', (lb1[:,:,0]*255).astype(np.uint8))

    mlflow_client.log_artifact(run_id, 'pred.jpg', 'runs/'+str(st))
    mlflow_client.log_artifact(run_id, 'imag.jpg', 'runs/'+str(st))
    mlflow_client.log_artifact(run_id, 'lbl.jpg', 'runs/'+str(st))
    
    mlflow_client.log_metric(run_id, 'train_loss', train_loss.result().numpy())
    mlflow_client.log_metric(run_id, 'train_iou', train_iou.result().numpy()*100)
    mlflow_client.log_metric(run_id, 'test_loss', test_loss.result().numpy())
    mlflow_client.log_metric(run_id, 'test_iou', test_iou.result().numpy()*100)

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_iou.reset_states()
    test_iou.reset_states()