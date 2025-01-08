import os
import shutil
import random
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

###########################
## PREPROCESSING DATASET ##
###########################

def merge_folders(folders_to_merge, output_folder, file_types):
    
    os.makedirs(output_folder, exist_ok=True)
    for folder in folders_to_merge:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path) and any(file_name.endswith(ext) for ext in file_types):
                dest_path = os.path.join(output_folder, file_name)
                shutil.copy(file_path, dest_path)
    print("Folders have been successfully merged!")

def reorganise_dataset(images_path, mask_path, output_folder, val_size=0.2, random_seed=0, silence=True):
    
    os.makedirs(os.path.join(output_folder, 'train_images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'train_masks', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val_images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val_masks', 'val'), exist_ok=True)

    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    random.shuffle(image_files)
    train_files = image_files[int(val_size * len(image_files)):]
    val_files = image_files[:int(val_size * len(image_files))]

    def move_file(image_filename, is_train):
        
        image_path = os.path.join(images_path, image_filename)
        tif_filename = image_filename.replace('.png', '_root_mask.tif')
        tif_path = os.path.join(mask_path, tif_filename)

        if os.path.exists(image_path) and os.path.exists(tif_path):
            
            if is_train:
                image_dest = os.path.join(output_folder, 'train_images', 'train', image_filename)
                mask_dest = os.path.join(output_folder, 'train_masks', 'train', tif_filename)
            else:
                image_dest = os.path.join(output_folder, 'val_images', 'val', image_filename)
                mask_dest = os.path.join(output_folder, 'val_masks', 'val', tif_filename)
            
            shutil.copy(image_path, image_dest)
            shutil.copy(tif_path, mask_dest)
            if not silence:
                print(f"Copied {image_filename} and {tif_filename} to {'train' if is_train else 'val'}")

        else:
            
            missing_files = []
            if not os.path.exists(image_path):
                missing_files.append(f"Image: {image_path}")
            if not os.path.exists(tif_path):
                missing_files.append(f"Mask: {tif_path}")
            if not silence:
                print(f"Skipping {image_filename} - Missing files: {', '.join(missing_files)}")

    for img_file in train_files:
        move_file(img_file, is_train=True)
    for img_file in val_files:
        move_file(img_file, is_train=False)
    print('Files have been successfully moved!')

    
def padder(image, patch_size):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become
    divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and
    right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to
    black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be
    divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding,
                                      left_padding, right_padding, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])

    return padded_image


def clean_images_and_masks(original_folder_path, dataset_type, padding=-10):
    """
    Cleans images and their corresponding masks by applying Gaussian blur, edge detection,
    and contour extraction, then cropping and resizing the images and masks into square images.
    Saves the cleaned images and masks into a structured directory.

    Parameters:
    - dataset_type (str): The type of the dataset to process (e.g., 'train', 'test').
    - padding (int): The amount of padding around the detected edges.

    Returns:
    None. The function saves the cleaned images and masks into a folder structure.
    """
    # Create a directory
    clean_dir = original_folder_path + '_clean'
    for subdir in ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val']:
        os.makedirs(os.path.join(clean_dir, subdir), exist_ok=True)
    
    for image_path in glob.glob(f'{original_folder_path}/{dataset_type}_images/{dataset_type}/*.png'):
        mask_path = image_path.replace('images', 'masks').replace('.png', '_root_mask.tif')

        image, mask = cv2.imread(image_path, 0), cv2.imread(mask_path, 0)

        cleaned_image, cleaned_mask, crop_coords = clean_image_and_mask(image, mask, padding=padding)

        # Ensure the cleaned image and mask have the same dimensions
        if cleaned_image.shape != cleaned_mask.shape:
            print(f"Image and mask dimensions do not match: {image_path}")

        # Save cleaned image
        cleaned_image_path = image_path.replace(original_folder_path, clean_dir)
        cv2.imwrite(cleaned_image_path, cleaned_image)

        # Save cleaned mask
        cleaned_mask_path = mask_path.replace(original_folder_path, clean_dir)
        cv2.imwrite(cleaned_mask_path, cleaned_mask)


def clean_image_and_mask(image, mask, padding=-10):
    """
    Cleans an individual image and its corresponding mask by applying Gaussian blur, Canny edge detection,
    contour extraction, and cropping around the detected contours. Then resizes both the image and the mask 
    to a square canvas with matching dimensions.

    Parameters:
    - image (numpy.ndarray): The image to be cleaned.
    - mask (numpy.ndarray): The mask to be cleaned.
    - padding (int): The padding to be applied around the detected contour.

    Returns:
    - numpy.ndarray: The cleaned and cropped image.
    - numpy.ndarray: The cleaned and cropped mask.
    - tuple: The cropping coordinates (x_start, y_start, x_end, y_end).
    """
    im_blurred = cv2.GaussianBlur(image, (13, 13), 0) # gaussian blur
    edges = cv2.Canny(im_blurred, threshold1=30, threshold2=30) # Canny edge detection

    im_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    edges = cv2.dilate(edges, None, iterations=2) # dilate edges to ensure we capture contours closer to the image border
    contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # find contours
    
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100] # filter out small contours
    
    # Find the largest contour (closest to image border)
    if contours:
        outermost_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    else:
        return image, (0, 0, image.shape[1], image.shape[0])  # Return original image and full crop coordinates
    
    im_color = cv2.drawContours(im_color, [outermost_contour], contourIdx=-1, color=(0, 255, 0), thickness=20) # draw the largest contour on the image
    x, y, w, h = cv2.boundingRect(outermost_contour) # get bounding box for the contour

    # Apply padding to the bounding box
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Crop the image and mask
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # Ensure the cropped image is square
    max_size = max(w, h)

    square_image = np.zeros((max_size, max_size), dtype=np.uint8)
    square_mask = np.zeros((max_size, max_size), dtype=np.uint8)
    
    # Center the cropped image in the square canvas
    x_offset = (max_size - w) // 2
    y_offset = (max_size - h) // 2
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image
    square_mask[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_mask

    return square_image, square_mask, (x, y, x + w, y + h)


def create_and_save_patches(original_folder_path, dataset_type, patch_size, scaling_factor):
    """
    Splits images and their corresponding masks from a blood cell dataset into smaller patches and saves them.

    This function takes images and masks from a specified dataset type, scales them if needed, and then splits them into smaller patches.
    Each patch is saved as a separate file. This is useful for preparing data for tasks like image segmentation in machine learning.

    Parameters:
    - dataset_type (str): The type of the dataset to process (e.g., 'train', 'test'). It expects a directory structure like
    'blood_cell_dataset/{dataset_type}_images/{dataset_type}' for images and 'blood_cell_dataset/{dataset_type}_masks/{dataset_type}' for
    masks.
    - patch_size (int): The size of the patches to be created. Patches will be squares of size patch_size x patch_size.
    - scaling_factor (float): The factor by which the images and masks should be scaled. A value of 1 means no scaling.

    Returns:
    None. The function saves the patches as .png files in directories based on their original paths, but replacing 'blood_cell_dataset'
    with 'blood_cell_dataset_patched'.

    Note:
    - The function assumes a specific directory structure and naming convention for the dataset.
    """
    # Create a directory to store the patches
    patch_dir = original_folder_path + '_patched'
    for subdir in ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val']:
        os.makedirs(os.path.join(patch_dir, subdir), exist_ok=True)

    # Take all png or tif files from the orginal dataset
    black_count_list = []
    white_count_list = []
    for image_path in glob.glob(f'{original_folder_path}/{dataset_type}_images/{dataset_type}/*.png'):
        mask_path = image_path.replace('images', 'masks').replace('.png', '_root_mask.tif')

        mask = cv2.imread(mask_path, 0)  # read in grayscale
        unique_values, counts = np.unique(mask, return_counts=True)

        # Condition 1: Remove if mask has only 1 unique value
        if len(unique_values) == 1:
            print(f"Skipping {image_path} and its mask due to a single unique value in the mask.")
            continue

        # Condition 2: Remove if mask has 2 unique values but fewer than 100 white pixels
        if len(unique_values) == 2 and (counts[1] if 1 in unique_values else 0) < 100:
            print(f"Skipping {image_path} and its mask due to fewer than 100 white pixels.")
            continue

        # IMAGE
        image = cv2.imread(image_path, 0) # read
        image = padder(image, patch_size) # pad
        if scaling_factor != 1: # scale
            image = cv2.resize(image, (0,0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(image, (patch_size, patch_size), step=patch_size) # patchify
        patches = patches.reshape(-1, patch_size, patch_size) # reshape

        image_patch_path = image_path.replace(original_folder_path, patch_dir) # replace path by new folder path
        for i, patch in enumerate(patches): # save in corresponding folder
            image_patch_path_numbered = f'{image_patch_path[:-4]}_{i}.png'
            cv2.imwrite(image_patch_path_numbered, patch)

        # MASK
        mask = cv2.imread(mask_path, 0) # read
        unique_values, counts = np.unique(mask, return_counts=True)
        mask = padder(mask, patch_size) # pad
        if scaling_factor != 1: # scale
            mask = cv2.resize(mask, (0,0), fx=scaling_factor, fy=scaling_factor)

        patches = patchify(mask, (patch_size, patch_size), step=patch_size) # patchify
        patches = patches.reshape(-1, patch_size, patch_size, 1) # reshape

        mask_patch_path = mask_path.replace(original_folder_path, patch_dir) # replace path by new folder path
        black_count = 0
        white_count = 0
        for i, patch in enumerate(patches): # save in corresponding folder
            _, counts = np.unique(patch, return_counts=True)
            black_count += counts[0]
            try:
                white_count += counts[1]
            except:
                white_count += 0

            # convert mask values (0 and 1) to 0 and 255 for PNG visibility
            patch = (patch * 255).astype(np.uint8)  # scale the mask to 0-255 and convert to uint8

            mask_patch_path_numbered = f'{mask_patch_path[:-4][:-10]}_{i}.png'
            cv2.imwrite(mask_patch_path_numbered, patch)
        print(f'total unique values in mask_patches  of original mask: {len(unique_values)}, {[black_count, white_count]}')
        black_count_list.append(black_count)
        white_count_list.append(white_count)
    if dataset_type == 'train':
        print(f'TOTAL BLACK PIXELS: {sum(black_count_list)}, TOTAL WHITE PIXELS: {sum(white_count_list)}, ')
        print('This can be used to balance model training')


###################
## VISUALISATION ##
###################

def show_random_patch_with_valid_overlay(patch_dir):
    valid_mask_found = False
    attempts = 0

    while not valid_mask_found and attempts < 100:

        image_path = np.random.choice(glob.glob(f'{patch_dir}/train_images/train/*.png'))
        mask_path = image_path.replace('images', 'masks')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.imread(mask_path, 0)
        if np.sum(mask) > 0:
            valid_mask_found = True
        else:
            attempts += 1

    if not valid_mask_found:
        raise ValueError("No valid mask found in the dataset within 100 attempts.")

    image = cv2.imread(image_path)
    if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask: {image_path}, {mask_path}")
        
    mask_color = [1, 0, 0]
    alpha = 0.5  # transparency

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.zeros_like(image_rgb, dtype=float)
    overlay[mask == 255] = mask_color
    overlay_image = image_rgb + alpha * overlay
    overlay_image = np.clip(overlay_image, 0, 1)
    overlay_image = (overlay_image * 255).astype(np.uint8)

    image_dims = image.shape[:2]
    mask_dims = mask.shape
    overlay_dims = overlay_image.shape[:2]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f"Original Image\n{image_dims[0]}x{image_dims[1]}")
    plt.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.05, pad=0.05)

    im1 = ax[1].imshow(mask, cmap='gray')
    ax[1].set_title(f"Mask\n{mask_dims[0]}x{mask_dims[1]}")
    plt.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.05, pad=0.05)

    im2 = ax[2].imshow(overlay_image, cmap='gray')
    ax[2].set_title(f"Overlay\n{overlay_dims[0]}x{overlay_dims[1]}")
    plt.colorbar(im2, ax=ax[2], orientation='vertical', fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()

##############
## TRAINING ##
##############

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + smooth)


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Original Author: Sreenivas Bhattiprolu
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1]) # !
    model.summary()
    
    return model


def create_data_generators(patch_dir, patch_size, batch_size=16):
    def generator(path, color_mode):
        datagen = ImageDataGenerator(rescale=1./255)
        return datagen.flow_from_directory(
            path,
            target_size=(patch_size, patch_size),
            batch_size=batch_size,
            class_mode=None,
            color_mode=color_mode,
            seed=42
        )

    train_images = generator(f'{patch_dir}/train_images/train', color_mode='grayscale')
    train_masks = generator(f'{patch_dir}/train_masks/train', color_mode='grayscale')
    val_images = generator(f'{patch_dir}/val_images/val', color_mode='grayscale')
    val_masks = generator(f'{patch_dir}/val_masks/val', color_mode='grayscale')

    print(f" - Shape of train image batch: {train_images[0].shape}")
    print(f" - Shape of train mask batch: {train_masks[0].shape}")
    print(f" - Shape of val image batch: {val_images[0].shape}")
    print(f" - Shape of val mask batch: {val_masks[0].shape}")

    return zip(train_images, train_masks), zip(val_images, val_masks), train_images, val_images


def load_images_from_folder(folder, target_size, color_mode):
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Add all image formats you expect
    for filename in sorted(os.listdir(folder)):  # Sort to ensure images and masks match
        if not filename.lower().endswith(valid_extensions):
            continue  # Skip non-image files
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        if color_mode == 'grayscale':
            img = img.convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        images.append(np.array(img) / 255.0)  # Normalize to [0, 1]
    return np.array(images)


def create_dataset_with_batches(patch_dir, patch_size, batch_size=16):
    # Define paths for images and masks
    train_images_dir = os.path.join(patch_dir, 'train_images', 'train')
    train_masks_dir = os.path.join(patch_dir, 'train_masks', 'train')
    val_images_dir = os.path.join(patch_dir, 'val_images', 'val')
    val_masks_dir = os.path.join(patch_dir, 'val_masks', 'val')

    # Load all data into memory
    X_train = load_images_from_folder(train_images_dir, target_size=(patch_size, patch_size), color_mode='grayscale')
    y_train = load_images_from_folder(train_masks_dir, target_size=(patch_size, patch_size), color_mode='grayscale')
    X_val = load_images_from_folder(val_images_dir, target_size=(patch_size, patch_size), color_mode='grayscale')
    y_val = load_images_from_folder(val_masks_dir, target_size=(patch_size, patch_size), color_mode='grayscale')

    # Function to yield batches
    def batch_generator(X, y, batch_size):
        num_samples = X.shape[0]
        while True:  # Infinite generator
            indices = np.arange(num_samples)
            np.random.shuffle(indices)  # Shuffle data at the start of each epoch
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                yield X[batch_indices], y[batch_indices]

    # Create generators for train and validation
    train_generator = batch_generator(X_train, y_train, batch_size)
    val_generator = batch_generator(X_val, y_val, batch_size)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    return train_generator, val_generator, X_train, y_train


###################
## ROBOTICS & RL ##
###################

def find_limit(simulation, axis, direction, step_size=0.1, max_steps=1000, silence=False):
    """
    Move the pipette along a specified axis in a given direction until it can no longer move.

    Args:
    - axis: 0 for x, 1 for y, 2 for z
    - direction: 1 for positive, -1 for negative
    - step_size: Incremental velocity to apply in the given direction
    - max_steps: Maximum number of steps to attempt

    Returns:
    - The coordinate value where movement stops
    """
    velocity = [0, 0, 0]
    previous_position = [0, 0, 0]
    boundary_position = None
    for _ in range(max_steps):
        velocity[axis] = step_size * direction
        actions = [[velocity[0], velocity[1], velocity[2], 0]]
        state = simulation.run(actions)
        
        # Extract current position from the state
        first_robot_key = list(state.keys())[0]
        current_position = state[first_robot_key]['pipette_position']
        
        # Check if the pipette is still moving along the axis
        if current_position[axis] == previous_position[axis]:
            boundary_position = current_position[axis]
            simulation.reset(num_agents=1)
            if not silence:
                print(boundary_position)
            return boundary_position
        
        # Update the previous position
        previous_position = current_position

def calculate_working_envelope(sim):
    """
    Given a simulation object, this function finds the limits along each axis,
    and calculates the coordinates of the working envelope's 8 corners.
    
    Args:
    - sim (Simulation): The simulation instance to be used for limit calculations.
    
    Returns:
    - list: A list of the coordinates of the 8 corners of the working envelope.
    """
    # Find the limits
    x_min = find_limit(sim, axis=0, direction=-1)
    x_max = find_limit(sim, axis=0, direction=1)
    y_min = find_limit(sim, axis=1, direction=-1)
    y_max = find_limit(sim, axis=1, direction=1)
    z_min = find_limit(sim, axis=2, direction=-1)
    z_max = find_limit(sim, axis=2, direction=1)
    
    # Define the 8 corners of the cube
    corners = [
        # Bottom square
        [x_min, y_min, z_min], # Bottom-left-back
        [x_min, y_max, z_min], # Bottom-right-back
        [x_max, y_min, z_min], # Bottom-left-front
        [x_max, y_max, z_min], # Bottom-right-front
        # Top square
        [x_min, y_min, z_max], # Top-left-back
        [x_min, y_max, z_max], # Top-right-back
        [x_max, y_min, z_max], # Top-left-front
        [x_max, y_max, z_max]  # Top-right-front
    ]
    return corners