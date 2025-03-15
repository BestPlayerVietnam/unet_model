import tensorflow as tf
from load_data import load_imgs_to_tensor, load_masks_to_tensor, tensor_to_patches, mask_to_patches
from config import Config
import os


def data_generator(split, ann_path):
    path = f'{Config.ROOT_DIR}/{split}/'   
    for seq_folder in os.listdir(path):
        img_path = os.path.join(path, seq_folder, 'Images')
        mask_path = os.path.join(path, seq_folder, 'Labels')
        tensor_images = load_imgs_to_tensor(img_path)
        tensor_masks = load_masks_to_tensor(mask_path, ann_path)
    
        for img, mask in zip(tensor_images, tensor_masks):
            patches_img = tensor_to_patches(img, Config.PATCH_HEIGHT, Config.PATCH_WIDTH)
            patches_mask = mask_to_patches(mask, Config.PATCH_HEIGHT, Config.PATCH_WIDTH)
            
            for patch_img, patch_mask in zip(patches_img, patches_mask):
                patch_mask = tf.one_hot(patch_mask, Config.NUM_CLASSES, dtype=tf.uint8)
                yield patch_img, patch_mask


