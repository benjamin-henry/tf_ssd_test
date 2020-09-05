# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from build_blaze import build_model as BlazeNet
import tensorflow.keras.backend as K
import numpy as np

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_loss_function.blaze_loss import smooth_l1_loss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

from bounding_box_utils.bounding_box_utils import *
from data_aug.data_aug import *
from preprocess import customDatagen


classes = {"gun":1}

img_height = 224 # Height of the input images
img_width = 224 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_predictor_layers = 3
intensity_mean = None # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = None # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = len(classes) # Number of positive classes
scales = None # [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

model_path = "./make_blaze_test.h5"

model, predictor_sizes = BlazeNet(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=None,
                aspect_ratios_global=aspect_ratios,
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=intensity_mean,
                divide_by_stddev=intensity_range,
                n_predictor_layers=n_predictor_layers,
                return_predictor_sizes=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
model.summary()

# K.clear_session() # Clear previous models from memory.
# model_path = 'blase_ssd.h5'
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'compute_loss': ssd_loss.compute_loss})
# model.summary()

checkpoint_path = './Checkpoints_blaze/'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path+'ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename=checkpoint_path+'ssd7_training_log.csv',
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]

with open('./guns_dataset/guns_csv.csv','r') as f:
    csv_path_array = f.read().strip().split('\n')

with open('./guns_dataset/negatives_csv.csv','r') as f:
    csv_neg_path_array = f.read().strip().split('\n')

images_path = './guns_dataset/Images'
negatives_path = './guns_dataset/Negatives'

batch_size = 24
epochs = 5

predictor_sizes = [model.get_layer('classes_1').output_shape[1:3],
                   model.get_layer('classes_2').output_shape[1:3],
                   model.get_layer('classes_3').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    coords='minmax',
                                    normalize_coords=normalize_coords)


model.fit(customDatagen(csv_path_array, csv_neg_path_array, images_path, negatives_path, batch_size, ssd_input_encoder), steps_per_epoch=int(len(csv_path_array*4) / batch_size), epochs=epochs)

model.save('blase_ssd.h5')