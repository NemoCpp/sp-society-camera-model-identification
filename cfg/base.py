
# Default locations
TRAIN_FOLDER       = 'train'
EXTRA_TRAIN_FOLDER = 'flickr_images'
EXTRA_VAL_FOLDER   = 'val_images'
TEST_FOLDER        = 'test'
MODEL_FOLDER       = 'models'

# Crop size (as in patch size)
CROP_SIZE = 512

# Classes
CLASSES = [
    'HTC-1-M7',
    'iPhone-6',     
    'Motorola-Droid-Maxx',
    'Motorola-X',
    'Samsung-Galaxy-S4',
    'iPhone-4s',
    'LG-Nexus-5x', 
    'Motorola-Nexus-6',
    'Samsung-Galaxy-Note3',
    'Sony-NEX-7']

# Classes via secondary namings
EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]

# Image resolutions observed within each phone class
RESOLUTIONS = {
    0: [[1520,2688]], # flips
    1: [[3264,2448]], # no flips
    2: [[2432,4320]], # flips
    3: [[3120,4160]], # flips
    4: [[4128,2322]], # no flips
    5: [[3264,2448]], # no flips
    6: [[3024,4032]], # flips
    7: [[1040,780],  # Motorola-Nexus-6 no flips
        [3088,4130], [3120,4160]], # Motorola-Nexus-6 flips
    8: [[4128,2322]], # no flips 
    9: [[6000,4000]], # no flips
}

# Map image resolutions with classes
for class_id,resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[class_id] = resolutions


# List of manipulations to be performed randomly
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

# Transformations for data augmentation during validation
VALIDATION_TRANSFORMS = [ [], ['orientation'], ['manipulation'], ['orientation','manipulation']]

# Total number of classes
N_CLASSES = len(CLASSES)



