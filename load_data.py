import tensorflow as tf
import keras as K
import os
import json
import numpy as np



def load_imgs_to_tensor(image_path):
    t = []
    for file in os.listdir(image_path):
        img_path = os.path.join(image_path, file)
        image = K.utils.load_img(img_path)
        image_array = K.utils.img_to_array(image)
        image_tensor = tf.convert_to_tensor(image_array) / 255.0
        t.append(image_tensor)
        
    return t

def tensor_to_patches(frame, patch_height, patch_width):
    frame = tf.expand_dims(frame, axis=0)
    patches = tf.image.extract_patches(images = frame, 
                                       sizes=[1, patch_height, patch_width, 1],
                                       strides = [1, patch_height, patch_width, 1],
                                       rates= [1, 1, 1, 1],
                                       padding= 'VALID')
    patches = tf.reshape(patches, [-1, patch_height, patch_width, 3])
    
    return patches

def load_masks_to_tensor(mask_path, annotation_path):
    #read annotation json
    with open(annotation_path, 'r') as f:
        class_map = json.load(f)
    #convert mask pixels to classes
    def mask_to_classes(mask_path):
        mask = K.utils.load_img(mask_path)
        mask_array = K.utils.img_to_array(mask).astype(np.uint8)
        class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)

        for class_id, color in class_map.items():
            # Найти все пиксели, которые соответствуют этому цвету
            match = (mask_array == color).all(axis=-1)
            # Присвоить им значение класса
            class_mask[match] = int(class_id)
        return tf.convert_to_tensor(class_mask)
    m = []
    for file in os.listdir(mask_path):
        m_path = os.path.join(mask_path, file)
        m.append(mask_to_classes(m_path))
        
    return m


def mask_to_patches(mask, patch_height, patch_width):
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=0)
        patches = tf.image.extract_patches(images = mask, 
                                        sizes=[1, patch_height, patch_width, 1],
                                        strides = [1, patch_height, patch_width, 1],
                                        rates= [1, 1, 1, 1],
                                        padding= 'VALID')
        patches = tf.reshape(patches, [-1, patch_height, patch_width, 1])
        patches = tf.squeeze(patches, axis=-1)
        
        return patches
