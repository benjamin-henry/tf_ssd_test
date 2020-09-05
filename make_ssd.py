import tensorflow as tf
import numpy as np
from models.ssd_model import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from tensorflow.keras.optimizers import Adam


img_height = 224 # Height of the input images
img_width = 224 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 20 # Number of positive classes
n_predictor_layers = 2
normalize_coords = True

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    n_predictor_layers=n_predictor_layers,
                    normalize_coords = normalize_coords
                    )

adam = Adam(learning_rate=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

model.save('model_train_test.h5')

