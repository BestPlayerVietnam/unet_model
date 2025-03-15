import cv2 as cv
import numpy as np
import json
import tensorflow as tf
import os
import datetime
from keras import layers, models, backend as K
from config import Config
from generate_data import data_generator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from metrics import combined_loss




    
def model_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Энкодер
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(x1)
    p1 = layers.MaxPooling2D()(x1)
    p1 = layers.Dropout(0.1)(p1)

    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(x2)
    p2 = layers.MaxPooling2D()(x2)
    p2 = layers.Dropout(0.1)(p2)

    # Центральный блок
    x3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x3 = layers.Conv2D(128, 3, padding='same', activation='relu')(x3)
    x3 = layers.Dropout(0.1)(x3)

    # Декодер
    u2 = layers.UpSampling2D()(x3)
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.Dropout(0.1)(u2)

    u1 = layers.UpSampling2D()(u2)
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = layers.Concatenate()([u1, x1])
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = layers.Dropout(0.1)(u1)

    # Выходной слой
    outputs = layers.Conv2D(Config.NUM_CLASSES, 1, activation='softmax')(u1)

    model = models.Model(inputs, outputs)
    return model

def calculate_class_weights(class_distribution):
    # Ваше распределение классов
    class_distribution = np.array(class_distribution)
    # Расчет весов по формуле 1 / (sqrt(freq) + epsilon)
    epsilon = 1e-8
    weights = 1.0 / (np.sqrt(class_distribution) + epsilon)
    weights /= np.sum(weights)  # Нормализация
    return tf.cast(weights, tf.float32)


train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator('train', Config.ANNOTATION_PATH),
    output_signature=(
         tf.TensorSpec(shape=(Config.PATCH_HEIGHT, Config.PATCH_WIDTH, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(Config.PATCH_HEIGHT, Config.PATCH_WIDTH, Config.NUM_CLASSES), dtype=tf.uint8)
    )
).batch(Config.BATCH_SIZE).repeat()  

valid_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator('valid', Config.ANNOTATION_PATH),
    output_signature=(
         tf.TensorSpec(shape=(Config.PATCH_HEIGHT, Config.PATCH_WIDTH, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(Config.PATCH_HEIGHT, Config.PATCH_WIDTH, Config.NUM_CLASSES), dtype=tf.uint8)
    )
).batch(Config.BATCH_SIZE).repeat()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
class_weights = calculate_class_weights(Config.CLASS_DISTRIBUTION)

model = model_unet(input_shape=(Config.PATCH_HEIGHT, Config.PATCH_WIDTH, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=combined_loss(class_weights),
    metrics=['accuracy']
)

model.fit(
    train_dataset ,
    steps_per_epoch=180,   
    validation_data=valid_dataset,
    validation_steps=50,   
    epochs=Config.EPOCHS,   
    callbacks=[reduce_lr, early_stopping]
)