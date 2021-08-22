# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:36:55 2021

@author: tboonesifuentes
"""

from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Activation, Lambda, Conv2D, MaxPool2D,AveragePooling2D, \
GlobalAveragePooling2D, Multiply, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19,VGG16
from keras import initializers
#from deakin.edu.au.data import Cifar100
import metrics as metrics
import numpy as np


class performance_callback(keras.callbacks.Callback):
    def __init__(self, X, y, taxo):
        self.X = X
        self.y = y
        self.taxo = taxo

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_pred = get_pred_indexes(y_pred)
        accuracy = metrics.get_accuracy(y_pred, self.y)
        exact_match = metrics.get_exact_match(y_pred, self.y)
        consistency = metrics.get_consistency(y_pred, self.taxo)
        print('-' * 100)
        print(f"epoch={epoch + 1}, ", end='')
        print(f"Exact Match = {exact_match:.4f}, ", end='')
        for i in range(len(accuracy)):
            print(f"accuracy level_{i} = {accuracy[i]:.4f}, ", end='')
        print(f"Consistency = {consistency:.4f}")
        print('-' * 100)
        print('')


def get_pred_indexes(y_pred):
    y_pred_indexes = []
    for pred in y_pred:
        
        y_pred_indexes.append(np.argmax(pred, axis=1))
       #y_pred_indexes.append(np.argmax(pred, axis=0))
       
    return y_pred_indexes


def get_mout_model(num_classes: list, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
                   learning_rate=1e-5, loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))

    # Build the model
    model = Model(name='mout_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_BCNN1(num_classes: list, image_size, reverse=False, conv_base=VGG19(include_top=False, weights="imagenet"),
              learning_rate=1e-5, loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    if reverse:
        num_classes = list(reversed(num_classes))
    for idx, v in enumerate(num_classes):
        if reverse:
            idx = len(num_classes) - idx - 1
        if len(out_layers) == 0:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))
        else:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(out_layers[-1]))
    if reverse:
        out_layers = list(reversed(out_layers))
    # Build the model
    model = Model(name='Model_BCNN1_reversed_' + str(reverse),
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_BCNN2(num_classes: list, image_size, reverse=False, conv_base=VGG19(include_top=False, weights="imagenet"),
              learning_rate=1e-5, loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    logits_layers = []
    out_layers = []
    if reverse:
        num_classes = list(reversed(num_classes))
    for idx, v in enumerate(num_classes):
        if reverse:
            idx = len(num_classes) - idx - 1
        if len(logits_layers) == 0:
            logits = Dense(v, name='logits_level_' + str(idx))(conv_base)
            out_layers.append(Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(Activation(keras.activations.relu)(logits))
        else:
            logits = Dense(v, name='logits_level_' + str(idx))(logits)
            out_layers.append(Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(Activation(keras.activations.relu)(logits))

    if reverse:
        out_layers = list(reversed(out_layers))
    # Build the model
    model = Model(name='Model_BCNN2_reversed_' + str(reverse),
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_mnets(num_classes: list, image_size, reverse=False, conv_base=[],
              learning_rate=1e-5, loss_weights=[]):
    in_layer = Input(shape=image_size, name='main_input')
    # Conv base
    if len(conv_base) == 0 or len(num_classes) != len(conv_base):
        conv_base = [VGG19(include_top=False, weights="imagenet") for x in num_classes]
    out_layers = []
    for i in range(len(conv_base)):
        conv_base[i]._name = 'conv_base' + str(i)
        conv_base[i] = conv_base[i](in_layer)
        conv_base[i] = Flatten()(conv_base[i])
        out_layers.append(Dense(num_classes[i], activation="softmax", name='out_level_' + str(i))(conv_base[i]))

    # Build the model
    model = Model(name='mnets',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


class BaselineModel(Model):
    def __init__(self, taxonomy, *args, **kwargs):
        super(BaselineModel, self).__init__(*args, **kwargs)
        self.taxonomy = taxonomy

    def predict(self, X):
        pred = super().predict(X)
        out = []
        for i in range(len(self.taxonomy) + 1):
            out.append([])
        for v in pred:
            child = np.argmax(v)
            out[-1].append(v)
            for i in reversed(range(len(self.taxonomy))):
                m = self.taxonomy[i]
                row = list(np.transpose(m)[child])
                parent = row.index(1)
                child = parent
                one_hot = np.zeros(len(row))
                one_hot[child] = 1
                out[i].append(one_hot)
        return out


def get_Baseline_model(num_classes: list, image_size, taxonomy, conv_base=VGG19(include_top=False, weights="imagenet"),
                       learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layer = Dense(num_classes[-1], activation="softmax", name='output')(conv_base)
    # Build the model
    model = BaselineModel(taxonomy=taxonomy, name='baseline_model',
                          inputs=in_layer,
                          outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_Classifier_model(num_classes, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
                         learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layer = Dense(num_classes, activation="softmax", name='output')(conv_base)
    # Build the model
    model = Model(name='simple_classifer',
                  inputs=in_layer,
                  outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_MLPH_model(num_classes: list, image_size, learning_rate=1e-5, loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = VGG19(include_top=False, weights="imagenet")
    # conv_base.summary()
    layer_outputs = [conv_base.get_layer("block5_conv1").output,
                     conv_base.get_layer("block5_conv2").output,
                     conv_base.get_layer("block5_conv3").output]
    conv_base_model = Model(inputs=conv_base.input, outputs=layer_outputs)
    conv_base = conv_base_model(in_layer)
    relu5_1_X = Flatten()(conv_base[0])
    relu5_2_Y = Flatten()(conv_base[1])
    relu5_3_Z = Flatten()(conv_base[2])
    UTX = Dense(512, activation="relu", name='UTX')(relu5_1_X)
    VTY = Dense(512, activation="relu", name='VTY')(relu5_2_Y)
    WTZ = Dense(512, activation="relu", name='WTZ')(relu5_3_Z)
    UTXVTY = Multiply(name='UTXVTY')([UTX, VTY])
    UTXWTZ = Multiply(name='UTXWTZ')([UTX, WTZ])
    VTYWTZ = Multiply(name='VTYWTZ')([VTY, WTZ])
    UTXVTYWTZ = Multiply(name='UTXVTYWTZ')([UTX, VTY, WTZ])
    concat = Concatenate()([UTXVTY, UTXWTZ, VTYWTZ, UTXVTYWTZ])
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(concat))

    # Build the model
    model = Model(name='MLPH_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model

def get_hdcnn_coarse_classifier(num_classes: list, image_size, level, learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    
    model = VGG16(include_top=False, weights="imagenet",input_tensor=in_layer)
    
    #Free parameters
    for i in range(len(model.layers)):
        
        model.layers[i].trainable=False
        
    x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(model.layers[-6].output)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_coarse = Dense(num_classes[level], activation='softmax',name='predictions')(x)
    
    # Build the model
    model_c = Model(name='hdcnn_model',inputs=in_layer,outputs=out_coarse)
    
    for i in range(len(model_c.layers)-5):
        model_c.layers[i].set_weights(model.layers[i].get_weights())
        
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model_c.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    
    return model_c

def get_hdcnn_fine_classifiers(num_classes: list, image_size, level, learning_rate=1e-5):
    
    in_layer = Input(shape=image_size, name='main_input')
    
    model = VGG16(include_top=False, weights="imagenet",input_tensor=in_layer)
    
    #Free parameters
    for i in range(len(model.layers)):
        
        model.layers[i].trainable=False
        
    x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(model.layers[-6].output)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_fine = Dense(num_classes[level], activation='softmax',name='predictions')(x)


    model_fine = Model(inputs=in_layer,outputs=out_fine)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model_fine.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    
    for i in range(len(model_fine.layers)-5):
        model_fine.layers[i].set_weights(model.layers[i].get_weights())
        
    return model_fine
