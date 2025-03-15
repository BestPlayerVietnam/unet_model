class Config:
    #IMG PARAMETERS
    IMG_HEIGHT = 720
    IMG_WIDTH = 1280
    PATCH_HEIGHT = IMG_HEIGHT // 4
    PATCH_WIDTH = IMG_WIDTH // 4
    ORIG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    PATCH_SIZE = (PATCH_WIDTH, PATCH_HEIGHT)
    NUM_CLASSES = 8
    CLASS_DISTRIBUTION = [
                        0.171, 0.304, 0.143, 0.26, 
                        0.094, 0.011, 0.014, 0.0016
                    ]
    
    
    #PATH PARAMETERS
    ROOT_DIR = 'src_hd/dataset'
    TRAIN_DIR = 'src_hd/train'
    VALID_DIR = 'src_hd/valid'
    TEST_DIR = 'src_hd/test'
    SAVE_DIR = 'models/unet.h5'
    LOGS_DIR = 'logs'
    ANNOTATION_PATH = 'annotation.json'
    
    #LEARN PARAMETERS
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4