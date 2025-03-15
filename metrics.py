import tensorflow as tf

def dice_coef(y_true, y_pred, class_weights, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    weighted_intersection = tf.reduce_sum(class_weights * y_true * y_pred, axis=[1, 2, 3])
    weighted_union = tf.reduce_sum(class_weights * y_true, axis=[1, 2, 3]) + tf.reduce_sum(class_weights * y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2. * weighted_intersection + smooth) / (weighted_union + smooth), axis=0)

def dice_loss(y_true, y_pred, class_weights):
    return 1 - dice_coef(y_true, y_pred, class_weights)

def focal_loss(y_true, y_pred, gamma=2.0, class_weights=None):
    # Приводим y_true к float32
    y_true = tf.cast(y_true, tf.float32)
    
    # Вычисляем кросс-энтропию
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Вычисляем вероятности
    pt = tf.math.exp(-ce)
    
    # Focal Loss с весами классов
    if class_weights is not None:
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        fl = weights * (1 - pt)**gamma * ce
    else:
        fl = (1 - pt)**gamma * ce
        
    return tf.reduce_mean(fl)

def combined_loss(class_weights):
    def loss_function(y_true, y_pred, alpha=0.7, beta=0.3, gamma=2.0):
        fl = focal_loss(y_true, y_pred, gamma=gamma, class_weights=class_weights)
        dl = dice_loss(y_true, y_pred, class_weights)
        return alpha * fl + beta * dl
    return loss_function