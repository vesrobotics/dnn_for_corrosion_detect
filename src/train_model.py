# for data load
import os

# for reading and processing images
import imageio
from PIL import Image

# for visualizations
import matplotlib.pyplot as plt

import numpy as np # for using np arrays

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

import argparse
import json

_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
_metrics = ['accuracy'] #, 'true_positives', 'true_negatives', 'false_positives', 'false_negatives', 'precision', 'recall']

def parse_args():
    print('Parsing data')
    parser = argparse.ArgumentParser(description = 'Parse input json')

    parser.add_argument('--path_to_params', type=str,
                        help='Path to a json file containing all of the training params required',
                        required=True)
    return parser.parse_args()

def json_params_parser(args):
    print('Parsing json')
    assert args.path_to_params is not None, \
        "Path to params for training is required with --path_to_params"
    
    with open(args.path_to_params) as json_file:
        data = json.load(json_file)

    return data

def LoadData (path1, path2):
    print('Loading images')
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    
    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        orig_img.append(file)
    for file in mask_dataset:
        mask_img.append(file)

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()
    
    return orig_img, mask_img

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2, startWithOne = False):
    print('Preprocessing images')
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
    y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    
    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        X[index] = single_img
        
        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
        if startWithOne : 
          single_mask = single_mask - 1
        y[index] = single_mask
    return X, y

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True, batch_norm = True, num_conv_layers = 2, add_max_pool_stride = False):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    if num_conv_layers == 3 :
      conv = Conv2D(n_filters, 
                    3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    if batch_norm:
      conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    max_pooling_stride = None
    if add_max_pool_stride : 
      max_pooling_stride = 2
      
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = max_pooling_stride)(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, batch_norm = False, num_conv_layers = 2, simple_upsampling = False, conv_stride_two = False):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """

    if simple_upsampling:
      up = UpSampling2D(size = (2,2))(prev_layer_input)
    else:
      # Start with a transpose convolution layer to first increase the size of the image
      up = Conv2DTranspose(
                  n_filters,
                  (3,3),    # Kernel size
                  strides=(2,2),
                  padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder

    relu = None
    if batch_norm == False : 
      relu = 'relu'

    conv_stride = (1,1)
    if conv_stride_two : 
      conv_stride = 2

    conv = Conv2D(n_filters, 
                  3,     # Kernel size
                  activation= relu,
                  padding='same',
                  strides= conv_stride,
                  kernel_initializer='HeNormal')(merge)
    if batch_norm:
      conv = BatchNormalization()(conv, training=False)
      n_filters = n_filters / 2

    if num_conv_layers == 2 :
      conv = Conv2D(n_filters,
                    3,   # Kernel size
                    activation= relu,
                    padding='same',
                    strides= conv_stride,
                    kernel_initializer='HeNormal')(conv)

      if batch_norm:
        conv = BatchNormalization()(conv, training=False)

    return conv

def UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output 
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)
    
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

def VggUNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output 
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)
    
    # Encoder 
    cblock1 = EncoderMiniBlock(inputs,      n_filters,    dropout_prob=0, max_pooling=True, batch_norm = False, add_max_pool_stride = True)
    cblock2 = EncoderMiniBlock(cblock1[0],  n_filters*2,  dropout_prob=0, max_pooling=True, batch_norm = False, add_max_pool_stride = True)
    cblock3 = EncoderMiniBlock(cblock2[0],  n_filters*4,  dropout_prob=0, max_pooling=True, batch_norm = False, num_conv_layers = 3, add_max_pool_stride = True)
    cblock4 = EncoderMiniBlock(cblock3[0],  n_filters*8,  dropout_prob=0, max_pooling=True, batch_norm = False, num_conv_layers = 3, add_max_pool_stride = True)
    cblock5 = EncoderMiniBlock(cblock4[0],  n_filters*8,  dropout_prob=0, max_pooling=True, batch_norm = False, num_conv_layers = 3, add_max_pool_stride = True) 
    
    # Decoder 
    ublock6 = DecoderMiniBlock(cblock5[0],  cblock4[0],  n_filters * 8, batch_norm = True, simple_upsampling = True, conv_stride_two = False)
    ublock7 = DecoderMiniBlock(ublock6,     cblock3[0],  n_filters * 4, batch_norm = True, simple_upsampling = True, conv_stride_two = False)
    ublock8 = DecoderMiniBlock(ublock7,     cblock2[0],  n_filters * 2, batch_norm = True, simple_upsampling = True, conv_stride_two = False)
    ublock9 = DecoderMiniBlock(ublock8,     cblock1[0],  n_filters,     batch_norm = True, simple_upsampling = True, num_conv_layers = 1, conv_stride_two = False)

    # Last Block
    up = UpSampling2D(size = (2,2))(ublock9)

    conv10 = Conv2D(n_classes,
                    3,
                    activation=None,
                    padding='same',
                    strides = 1,
                    kernel_initializer='he_normal')(up)

    conv10 = BatchNormalization()(conv10, training=False)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

def show_result_training(results):
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label = 'train loss')
    axis[0].plot(results.history["val_loss"], color='b', label = 'dev loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(results.history["accuracy"], color='r', label = 'train accuracy')
    axis[1].plot(results.history["val_accuracy"], color='b', label = 'dev accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()
    return

def train_model(args):

    data = json_params_parser(args)

    if not data:
        print("Json is empty!")
        return False

    path_to_raw_data = data['path_to_raw_data']
    path_to_mask_data = data['path_to_mask_data']
    path_to_saved_weights = data['path_to_saved_weights']
    target_shape_img = data['target_img_shape']
    target_shape_mask = data['target_mask_shape']
    n_filters = data['n_filters']
    n_classes = data['n_classes']
    test_size = data['test_size']
    batch_size = data['batch_size']
    epochs = data['epochs']
    model_to_use = data['model_to_use']

    img, mask = LoadData (path_to_raw_data, path_to_mask_data)
    # Process data using apt helper function
    X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path_to_raw_data, path_to_mask_data)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=123)
    assert not np.any(np.isnan(X_train)), \
        "Preprocessing failed"
    assert not np.any(np.isnan(X_valid)), \
        "Preprocessing failed"
    assert not np.any(np.isnan(y_train)), \
        "Preprocessing failed"
    assert not np.any(np.isnan(y_valid)), \
        "Preprocessing failed"

    if model_to_use == 0 :
        # Call the helper function for defining the layers for the model, given the input image size
        model = UNetCompiled(input_size=(target_shape_img[0],target_shape_img[1],target_shape_img[2]), n_filters=n_filters, n_classes=n_classes)
    elif model_to_use == 1 : 
        model = VggUNetCompiled(input_size=(target_shape_img[0],target_shape_img[1],target_shape_img[2]), n_filters=n_filters, n_classes=n_classes)
    else:
        print('--model_to_use has to be 0 for unet or 1 for vgg-unet')
        return False

    # Check the summary to better interpret how the output dimensions change in each layer
    model.summary()

    # There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
    # Ideally, try different options to get the best accuracy
    model.compile(  optimizer=_optimizer, 
                    loss=_loss,
                    metrics=_metrics)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(   filepath=path_to_saved_weights,
                                                        save_weights_only=True,
                                                        verbose=1)


    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[cp_callback])

    show_result_training(results)

    return True

if __name__ == '__main__':
    args = parse_args()
    if train_model(args):
        print("Execution succeed :)")
    else:
        print("Execution caca :(")
