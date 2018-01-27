
import random

# Make random behaviour repeatable for debugging purposes
# Do this first, as any other module that includes random will now be
# pre-seeded unless it specifically changes the seed itself.
SEED=42
random.seed(SEED)

import argparse
import re
import sys
import os
from os.path import join
import glob

import numpy as np
import math

from io import BytesIO
import copy

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam, Adadelta, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, SeparableConv2D, Reshape
from keras.utils import to_categorical, Sequence
from keras.initializers import Constant
from keras.applications import *

from lib.inputgenerator import InputGenerator

# Import our configuration
from cfg.base import *

# Import our class helper
from lib.classhelper import *

# Process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for FC layers')
parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use')
parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')
parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')

args = parser.parse_args()

# Modify any configuration variables that need adjusting based on CLI arguments
CROP_SIZE = args.crop_size


# MAIN

# Load or create a new model based on CLI arguments
if args.model:
    # Load the model avoiding compilation since that will be done at a later
    # stage
    print("Loading model " + args.model)
    model = load_model(args.model, compile=False)

    # Get the model name and epoch number from the filename
    match = re.search(r'([A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(2))
else:
    # Brand new model
    last_epoch = 0

    # Initial images and manipulation arrays
    input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3))
    manipulated = Input(shape=(1,))

    # Get the classifier name and it's model
    classifier = globals()[args.classifier]

    classifier_model = classifier(
        include_top=False, 
        weights = 'imagenet' if args.use_imagenet_weights else None,
        input_shape=(CROP_SIZE, CROP_SIZE, 3), 
        pooling=args.pooling if args.pooling != 'none' else None)

    # start with the input preparation
    x = input_image
    # Add kernel filter if needed. Note the padding
    if args.learn_kernel_filter:
        x = Conv2D(3, (7, 7), strides=(1,1), use_bias=False, padding='valid', name='filtering')(x)

    # Add the pre-defined classifier network 
    x = classifier_model(x)

    # Get the output of the model so far and add the manipulation flag to
    # prepare for the FC/Softmax classifier stages. After the reshape and
    # contatenation we should have a flatten array that can be used directly
    x = Reshape((-1,))(x)
    if args.dropout_classifier != 0.:
        x = Dropout(args.dropout_classifier, name='dropout_classifier')(x)
    x = concatenate([x, manipulated])

    # Add the fully connected layers unless told not to
    if not args.no_fcs:
        # First fully connected layer and a dropout to reduce overfitting
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(args.dropout,         name='dropout_fc1')(x)
        # Same thing for the second fully connected layer
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(args.dropout,         name='dropout_fc2')(x)

    # Add the prediction layer
    prediction = Dense(N_CLASSES, activation ="softmax", name="predictions")(x)

    # We are done with the model architecture
    model = Model(inputs=(input_image, manipulated), outputs=prediction)

    # Create an informative name for our model based on the parameters used
    model_name = args.classifier + \
        ('_kf' if args.kernel_filter else '') + \
        ('_lkf' if args.learn_kernel_filter else '') + \
        '_do' + str(args.dropout) + \
        '_' + args.pooling

    # Load the weights if needed
    if args.weights:
            model.load_weights(args.weights, by_name=True)
            match = re.search(r'([A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.weights)
            #model_name = match.group(1)
            last_epoch = int(match.group(2))

# Show a summary of our model
model.summary()

# Split the model across GPUs
if args.gpus > 1:
    from lib.multi_gpu_keras import multi_gpu_model
    model = multi_gpu_model(model, gpus=args.gpus)

#if not (args.test or args.test_train):
# TRAINING

# Get a list of files
ids = glob.glob(join(TRAIN_FOLDER,'*/*.jpg'))
ids.sort()

# If that's all our data, go ahead and do a train/test split from it
if not args.extra_dataset:
    # Here we do need to force the seed if we want reproducible random Sequences
    ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)
else:
    # Add Gleb's list of images from flickr
    ids_train = ids
    ids_val   = [ ]

    # First add the train images
    extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
    extra_train_ids.sort()
    ids_train.extend(extra_train_ids)

    # Now create the validation set
    extra_val_ids = glob.glob(join(EXTRA_VAL_FOLDER,'*/*.jpg'))
    extra_val_ids.sort()
    ids_val.extend(extra_val_ids)

# Get the classes from the filenames
classes = [get_class_id(idx.split('/')[-2]) for idx in ids_train]

# Create a class weight to make up for the unbalance (this is important when using
# the flickr images. The training images from the competition are balanced)
class_weight = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

# Show info about the classes and how files are split among them
classes_count = np.bincount(classes)
print("\nImages per class breakup and class weights\n")
for class_name, class_count, weight in zip(CLASSES, classes_count,class_weight):
    print('{:>22}: {:5d} ({:04.1f}%) {:4.3f}'.format(
        class_name, class_count, 100. * class_count / len(classes), weight))
print("\n{:>22}: {:5d}\n".format("Total",classes_count.sum()))

# Initialize the input generator
generator = InputGenerator(train=ids_train,val=ids_val,bs=args.batch_size,crop_size=CROP_SIZE)
train_generator = generator.train_generate()
val_generator = generator.val_generate()


# Get the optimizer
opt = Adam(lr=args.learning_rate)
#opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the model
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# We'll be monitoring the validation accuracy and use it for saving checkpoints
# and automatically reducing the learning rate
monitor = 'val_acc'

# We also want to include the validation accuracy in the filename when saving checkpoints
# to make it easy to figure out what checkpoint to use
metric  = "-val_acc{val_acc:.6f}"

# Create the callback for saving checkpoints. For now we'll just save them
# when the validation accuracy improves
save_checkpoint = ModelCheckpoint(
    join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}"+metric+".hdf5"),
    monitor=monitor, save_best_only=True, save_weights_only=False, period=1,
    verbose=0, mode='max')

# Create the callback for automatically reducing the learning rate
reduce_lr = ReduceLROnPlateau(
    monitor=monitor,factor=0.5, patience=10,min_lr=1e-9, epsilon = 0.00001,
    verbose=1, mode='max')

# Go ahead and do the training
model.fit_generator(
        generator        = train_generator,
        steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
        validation_data  = val_generator,
        validation_steps = int(len(VALIDATION_TRANSFORMS) * math.ceil(len(ids_val) // args.batch_size)),
        epochs = args.max_epoch,
        callbacks = [save_checkpoint, reduce_lr],
        initial_epoch = last_epoch,
        max_queue_size = 10,
        class_weight=class_weight)
