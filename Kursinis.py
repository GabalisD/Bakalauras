import json, h5py, os, shutil, sys
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image

import tensorflow as tf
from tensorflow import saved_model
from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework import ops
import tensorflow_addons as tfa

import math
import os

batch_size = 1
argN = 64

train_folder = '/content/drive/My Drive/KursinisData/data'
test_folder = '/content/drive/My Drive/KursinisData/data'
test_transformed ='/content/drive/My Drive/KursinisData'

train_dir_anchor = os.path.join(train_folder, 'train_a')
train_dir_positive = os.path.join(train_folder, 'train_p')
train_dir_negative = os.path.join(train_folder, 'train_n')

valid_dir_anchor = os.path.join(train_folder, 'valid_a')
valid_dir_positive = os.path.join(train_folder, 'valid_p')
valid_dir_negative = os.path.join(train_folder, 'valid_n')

test_dir_anchor = os.path.join(test_folder, 'test_a')
test_dir_positive = os.path.join(test_folder, 'test_p')
test_dir_negative = os.path.join(test_folder, 'test_n')

def train_generator_triplet():
    gen = ImageDataGenerator()
    gen_a = gen.flow_from_directory(directory = train_dir_anchor, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
    gen_p = gen.flow_from_directory(directory = train_dir_positive, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
    gen_n = gen.flow_from_directory(directory = train_dir_negative, target_size = (224, 224), batch_size = batch_size, class_mode = 'categorical', shuffle = False)
    while True:
        an = gen_a.next()
        po = gen_p.next()
        ne = gen_n.next()
        yield [an[0], po[0], ne[0]], an[1]

def valid_generator_triplet():
    gen = ImageDataGenerator()
    gen_a = gen.flow_from_directory(directory = valid_dir_anchor, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
    gen_p = gen.flow_from_directory(directory = valid_dir_positive, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
    gen_n = gen.flow_from_directory(directory = valid_dir_negative, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
    while True:
        an = gen_a.next()
        po = gen_p.next()
        ne = gen_n.next()
        yield [an[0], po[0], ne[0]], an[1]

    def test_generator_triplet():
        gen = ImageDataGenerator()
        gen_a = gen.flow_from_directory(directory = test_dir_anchor, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
        gen_p = gen.flow_from_directory(directory = test_dir_positive, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
        gen_n = gen.flow_from_directory(directory = test_dir_negative, target_size = (224, 224), batch_size = 1, class_mode = 'categorical', shuffle = False)
        while True:
            an = gen_a.next()
            po = gen_p.next()
            ne = gen_n.next()
            yield [an[0], po[0], ne[0]], an[1]

            def triplet_loss(N = argN, epsilon = 1e-6)

    def triplet_loss(y_true, y_pred):
        beta = N
        alpha = 0.5
        print("Shape:", y_pred.get_shape())

        anchor = y_pred[0::3]
        positive = y_pred[1::3]
        negative = y_pred[2::3]

        positive_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, positive), 2), 1, keepdims=True))
        negative_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, negative), 2), 1, keepdims=True))
        
        loss = tf.reduce_mean(tf.maximum(positive_distance - negative_distance + alpha, 0))

        return loss
    return triplet_loss

def pd(N = argN, epsilon = 1e-6):
    def pd(y_true, y_pred):
        beta = N
        anchor = y_pred[0::3]
        positive = y_pred[1::3]
        positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=0)
        return backend.mean(positive_distance)
    return pd

def nd(N = argN, epsilon = 1e-06):
    def nd(y_true, y_pred):
        beta = N
        anchor = y_pred[0::3]
        negative = y_pred[2::3]
        negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=0)
        return backend.mean(negative_distance)
    return nd

    def make_model():
    input_a = Input(shape = (224, 224, 3),  name = 'input_a')
    input_p = Input(shape = (224, 224, 3),  name = 'input_p')
    input_n = Input(shape = (224, 224, 3),  name = 'input_n')

    base_model = applications.VGG16(include_top = False, weights = 'imagenet')

    l1_a = base_model.layers[0](input_a)
    l1_p = base_model.layers[0](input_p)
    l1_n = base_model.layers[0](input_n)

    l2_a = base_model.layers[1](l1_a)
    l2_p = base_model.layers[1](l1_p)
    l2_n = base_model.layers[1](l1_n)

    l3_a = base_model.layers[2](l2_a)
    l3_p = base_model.layers[2](l2_p)
    l3_n = base_model.layers[2](l2_n)

    l4_a = base_model.layers[3](l3_a)
    l4_p = base_model.layers[3](l3_p)
    l4_n = base_model.layers[3](l3_n)

    l5_a = base_model.layers[4](l4_a)
    l5_p = base_model.layers[4](l4_p)
    l5_n = base_model.layers[4](l4_n)

    l6_a = base_model.layers[5](l5_a)
    l6_p = base_model.layers[5](l5_p)
    l6_n = base_model.layers[5](l5_n)

    l7_a = base_model.layers[6](l6_a)
    l7_p = base_model.layers[6](l6_p)
    l7_n = base_model.layers[6](l6_n)

    l8_a = base_model.layers[7](l7_a)
    l8_p = base_model.layers[7](l7_p)
    l8_n = base_model.layers[7](l7_n)


    lt1 = Dense(64, activation = 'sigmoid')
    lt2 = Dropout(0.5)
    lt3 = Dense(8, activation = 'sigmoid')

    lt1_a = lt1(l8_a)
    lt1_p = lt1(l8_p)
    lt1_n = lt1(l8_n)

    lt2_a = lt2(lt1_a)
    lt2_p = lt2(lt1_p)
    lt2_n = lt2(lt1_n)

    lt3_a = lt3(lt2_a)
    lt3_p = lt3(lt2_p)
    lt3_n = lt3(lt2_n)

    output = tf.keras.layers.concatenate([lt3_a, lt3_p, lt3_n], axis = 0, name = 'out666')
    model = tf.keras.models.Model(inputs = [input_a, input_p, input_n], outputs = output)

    for layer in model.layers:
        if layer.name == 'dense':
            break
        layer.trainable = False

    model.compile(optimizer = optimizers.Adam(), loss = triplet_loss(), metrics = [pd(), nd()])

    return model

    def plot_metrics(history, metrics=['loss'], skip_start=0.):
    """
    Plots metrics from keras training history.
    """
    hist = history.history
    start_indice = int(len(hist[metrics[0]]) * skip_start)
    
    for metric in metrics:
        plt.plot(hist[metric][start_indice:], label="train {}".format(metric))
        plt.plot(hist[f"val_{metric}"][start_indice:], label=f"val {metric}")
        plt.legend()
        plt.title(metric)
        plt.figure()
    
    plt.show()

    callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
]


model = make_model()
history = model.fit_generator(
    generator = train_generator_triplet(), 
    steps_per_epoch = 1188, 
    epochs = 16, 
    validation_data = valid_generator_triplet(), 
    validation_steps = 100, 
    callbacks = callbacks,
    shuffle = False
)

test_samples = len(os.listdir(os.path.join(test_dir_positive,"0")))

results = model.predict_generator(generator = test_generator_triplet(), steps = test_samples, verbose = 0)

results_images = results[0,:,:,:]
print(results_images.shape)
for i in range(8):
    plt.imshow(results_images[:,:,i])
    plt.show()

anchor = results[0::3]
positive = results[1::3]
negative = results[2::3]

def print_results(index, anchor, positive, negative):
    for embedding_index in range(8):
        print("Showing triplet embedding: {}".format(embedding_index))
        plt.imshow(anchor[index,:,:,embedding_index])
        plt.show()
        plt.imshow(positive[index,:,:,embedding_index])
        plt.show()
        plt.imshow(negative[index,:,:,embedding_index])
        plt.show()

print_results(0, anchor, positive, negative)


beta = argN
epsilon = 1e-6

anchor = results[0::3]
positive = results[1::3]
negative = results[2::3]

print(np.square(anchor - negative).shape)

positive_distance = np.nansum(np.square(anchor - positive), axis = 1)
negative_distance = np.nansum(np.square(anchor - negative), axis = 1)


tp = 0
fp = 0
pneq = 0
min_p = sys.maxsize
max_p = 0
min_n = sys.maxsize
max_n = 0


pd_list = []
nd_list = []

for i in range(test_samples):
    pda = np.nansum(positive_distance[i])
    nda = np.nansum(negative_distance[i])
    
    pd_list.append(pda)
    nd_list.append(nda)

    for pd, nd in zip(pd_list, nd_list):
      pda = (pd / 25088)
      nda = (nd / 25088)

    if pda >= 0.25:
        fp += 1
    else:
        tp += 1
    if pda == nda:
        pneq += 1

    if min_p > pda:
        min_p = pda
    if max_p < pda:
        max_p = pda

    if min_n > nda:
        min_n = nda
    if max_n < nda:
        max_n = nda 
        



print(min_p, ' - ', max_p, ', ', min_n, ' - ', max_n)
print('accuracy: ', np.round(tp / (tp + fp) * 100, 1))
print('equal predictions: ', pneq)

--------------------------image maker----------------------------------

from PIL import Image 
from PIL import ImageFilter 
import os, fileinput, sys

for entry in os.scandir('drive/My Drive/KursinisData/Test'): 
    if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        filename=entry.path
        (name, extension) = os.path.splitext(entry.path)

        img = img.filter(ImageFilter.BoxBlur(7))
        (name1,imgName) =filename.rsplit('/', 1)

        img.save('drive/My Drive/KursinisData/data/test_p/0/' + imgName)