####################################
# Image to Arrays
from tqdm import tqdm as tqdm
import numpy as np
import cv2 as cv
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import imgaug.augmenters as iaa
import math
import random
from random import randint

img_size = 224



def image_to_array_raw(image_filepaths, img_size):
    """"
    This code block is to take the images and transform them to arrays
    Raw, no pre-processing, no normalization of pixels

    Args:

    Returns: array of shape (len(image_filepaths), img_size, img_size, channels)  
    """
    X_dataset = []
    
    for i in tqdm(range(len(image_filepaths))):
        img = image.load_img('Images/'+image_filepaths[i], target_size = (img_size, img_size))
        img = image.img_to_array(img)
        X_dataset.append(img)
    X = np.array(X_dataset)

    return X

def image_to_array_norm_pix(image_filepaths, img_size):
    """"
    This code block is to take the images and transform them to arrays
    Normalization of pixels by dividing by 255, no other pre-processing

    Args:

    Returns: array of shape (len(image_filepaths), img_size, img_size, channels)  
    """
    X_dataset = []
    
    for i in tqdm(range(len(image_filepaths))):
        img = image.load_img('Images/'+image_filepaths[i], target_size = (img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        X_dataset.append(img)
    X = np.array(X_dataset)

    return X

def image_to_array_preresnet50(image_filepaths, img_size):
    """"
    This code block is to take the images and transform them to arrays
    No pixel normalization
    Only resnet50 pre-processing

    Args:

    Returns: array of shape (len(image_filepaths), img_size, img_size, channels)  
    """
    X_dataset = []
    
    for i in tqdm(range(len(image_filepaths))):
        img = image.load_img('Images/'+image_filepaths[i], target_size = (img_size, img_size))
        img = image.img_to_array(img)
        img = keras.applications.resnet50.preprocess_input(img)
        X_dataset.append(img)
    X = np.array(X_dataset)

    return X

def load_prep_img(image_path, target_shape=(img_size, img_size)):
    """"
    This code block is to take the images and transform them to arrays
    and to clip the black background

    Args: image filepath, image size

    Returns: one image, array of shape img_size, img_size, channels)  
    """

    image = cv.imread(image_path, cv.IMREAD_COLOR) # load from the directory
    non_0_rows = np.array([row_idx for row_idx, row in enumerate(image) if np.count_nonzero(row)!=0])
    non_0_cols = np.array([col_idx for col_idx, col in enumerate(image.transpose(1,0,2)) if np.count_nonzero(col)!=0])
    image = image[non_0_rows.min():non_0_rows.max()+1, non_0_cols.min():non_0_cols.max()+1, :] # clip
    image = cv.resize(image, target_shape)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # convert to RGB
    
    return image

def image_to_array_no_background(image_filepaths, img_size):
    """"
    This code block is to take the images and transform them to arrays
    Loops through the load_prep_image function
    Removes black background, but other than that its raw, no pre-processing, no normalization of pixels

    Args:

    Returns: array of shape (len(image_filepaths), img_size, img_size, channels)  
    """
    X_dataset = []
    
    for i in tqdm(range(len(image_filepaths))):
        img = load_prep_img('Images/'+image_filepaths[i], target_shape=(img_size, img_size))
        X_dataset.append(img)
    X = np.array(X_dataset)

    return X

####################################
# CLAHE
def apply_clahe_hsv(image, clip=1.0, grid_size=(8,8)):
    """
    This function applies CLAHE to a color image by converting RGB to HSV, 
    and then applying CLAHE to v, the brightness channel. Then the new HSV is
    converted back to an RGB image

    Args: image (array of shape 224 x 224 x 3 for example), 
    clip (cliplimit)
    grid_size (must be a tuple)
    """

    # Convert RGB to HSV (hue, saturation, brightness)
    hsv_img = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Save HSV channels into separate variables
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    # Apply CLAHE only to brightness channel v and save it as v
    v = clahe.apply(v)

    # Re-construct HSV image
    hsv_img = np.dstack((h,s,v))

    # Convert HSV to RGB
    rgb = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)

    return rgb

def apply_clahe_lab(image, clip=1.0, grid_size=(8,8)):
    """
    This function applies CLAHE to a color image by converting RGB to HSV, 
    and then applying CLAHE to v, the brightness channel. Then the new HSV is
    converted back to an RGB image

    Args: image (array of shape 224 x 224 x 3 for example), 
    clip (cliplimit)
    grid_size (must be a tuple)
    """

    # Convert RGB to HSV (hue, saturation, brightness)
    lab_img = cv.cvtColor(image, cv.COLOR_RGB2LAB)

    # Save HSV channels into separate variables
    l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    # Apply CLAHE only to light channel l and save it as l
    l = clahe.apply(l)

    # Re-construct HSV image
    lab_img = np.dstack((l,a,b))

    # Convert HSV to RGB
    rgb = cv.cvtColor(lab_img, cv.COLOR_LAB2RGB)

    return rgb

def apply_clahe_rgb(image, clip=1.0, grid_size=(8,8)):
    """
    This function applies CLAHE to a color image by applying CLAHE to RGB channels separately

    Args: image (array of shape 224 x 224 x 3 for example), 
    clip (cliplimit)
    grid_size (must be a tuple)

    https://www.freedomvc.com/index.php/2021/09/19/color-image-histogram-clahe/
    """

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    # Apply CLAHE only to RGB channels separately
    r = clahe.apply(image[:,:,0])
    g = clahe.apply(image[:,:,1])
    b = clahe.apply(image[:,:,2])

    # Re-construct RGB image
    rgb = np.dstack((r,g,b))

    return rgb

def apply_green_channel(image, clip=1.0, grid_size=(8,8), filtering=True, 
gaussian_grid=(3,3), gaussian_border=cv.BORDER_CONSTANT):
    """
    This function applies CLAHE to the green channel of an RGB image, 
    followed by Histogram Equalization and Gaussian filtering

    Args: image (array of shape 224 x 224 x 3 for example), 
    clip (cliplimit)
    grid_size (must be a tuple)

    https://www.freedomvc.com/index.php/2021/09/19/color-image-histogram-clahe/
    https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html

    returns: g (green channel with CLAHE, HE, and Gaussian applied)
    """

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    # Apply CLAHE only to only the green channel
    g = clahe.apply(image[:,:,1])

    # Histogram equalization
    processed = cv.equalizeHist(g)

    if filtering:
    # # Gaussian filtering
        processed = cv.GaussianBlur(processed, gaussian_grid, gaussian_border)


    return processed

def apply_green_CLAHE(image, clip=1.0, grid_size=(8,8)):
    """
    This function applies CLAHE to the green channel of an RGB image

    Args: image (array of shape 224 x 224 x 3 for example), 
    clip (cliplimit)
    grid_size (must be a tuple)

    https://www.freedomvc.com/index.php/2021/09/19/color-image-histogram-clahe/

    returns: g (green channel with CLAHE applied)
    """

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    # Apply CLAHE only to only the green channel
    g = clahe.apply(image[:,:,1])

    return g
##################
# Apply CLAHE to all
def apply_clahe_all(clahe_method, image_set, clip=1.0, grid_size=(8,8), filtering=True):
    X_clahe_list = []

    if clahe_method == apply_green_channel:
        for i in tqdm(range(len(image_set))):
            img = clahe_method(image_set[i], clip=clip, grid_size=grid_size, filtering=filtering)
            X_clahe_list.append(img)
    
    else:
        for i in tqdm(range(len(image_set))):
            img = clahe_method(image_set[i], clip=clip, grid_size=grid_size)
            X_clahe_list.append(img)
    
    X_clahe = np.array(X_clahe_list)

    return X_clahe


####################################
#ORB
def apply_orb(images, max_features=500, keep_descriptors=200):
    X_descriptors_list = []

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=max_features)

    for i in tqdm(range(len(images))):
        # find the keypoints with ORB
        kp = orb.detect(images[i], None)

        # compute the descriptors with ORB
        kp, des = orb.compute(images[i], kp)

        X_descriptors_list.append(des[0:keep_descriptors])
    
    X_descriptors = np.array(X_descriptors_list, dtype=object)

    return X_descriptors


####################################
# Calc Single Diagnosis and Subsets

normal = np.array([1, 0, 0, 0, 0, 0, 0, 0])
diabetes = np.array([0, 1, 0, 0, 0, 0, 0, 0])
glaucoma = np.array([0, 0, 1, 0, 0, 0, 0, 0])
cataracts = np.array([0, 0, 0, 1, 0, 0, 0, 0])
amd = np.array([0, 0, 0, 0, 1, 0, 0, 0])
hypertension = np.array([0, 0, 0, 0, 0, 1, 0, 0])
myopia = np.array([0, 0, 0, 0, 0, 0, 1, 0])
other = np.array([0, 0, 0, 0, 0, 0, 0, 1])

def calc_single_diagnosis(labels):
    # Counter
    n = 0
    d = 0
    g = 0
    c = 0
    a = 0
    h = 0
    m = 0
    o = 0 

    for label in labels:      
        if np.array_equal(label, normal):
            n+=1
        if np.array_equal(label, diabetes):
            d+=1
        if np.array_equal(label, glaucoma):
            g+=1
        if np.array_equal(label, cataracts):
            c+=1
        if np.array_equal(label, amd):
            a+=1
        if np.array_equal(label, hypertension):
            h+=1
        if np.array_equal(label, myopia):
            m+=1
        if np.array_equal(label, other):
            o+=1
    
    return [n, d, g, c, a, h, m, o]

def find_desired_indices(labels, selected_diseases):
    """
    Create a list of indices for samples if the following conditions are met

    Args: list of labels

    returns: a list of indices that point to the diseases we want
    """

    desired_indices_list = []

    for idx in range(len(labels)):
        # Check and add if only labeled as normal
        for selected_disease in selected_diseases:
            if np.array_equal(labels[idx], selected_disease):
                desired_indices_list.append(idx)
        
    return desired_indices_list

def create_data_subset(desired_indices_list, X, make_array=True):
    """
    Create a subset of the data from the list of indices

    Args: 
        X = dataset, 
        indices_list = list of indices of the samples we want

    returns: subset containing the data we want
    """

    desired_samples = []

    for idx in tqdm(desired_indices_list):
        desired_samples.append(X[idx])

    if make_array == True:
        X_subset = np.array(desired_samples)
    
    else:
        X_subset = desired_samples

    return X_subset

def create_subset_labels(desired_indices_list, labels):
    desired_labels = []

    for idx in tqdm(desired_indices_list):
        desired_labels.append(labels[idx])
    
    y_subset = np.array(desired_labels)

    return y_subset

def create_subsets(disease_vector_labels, df_disease_columns, labels, df, X):
    """Combine all subset creation functions into 1
    Args:
    disease_vector_labels = list of vector labels we wish to select
    df_disease_columns - list of column names of diseases
    labels - array of labels
    df - dataframe
    X - samples

    Returns:
    X_subset - contains samples of a single disease
    y_subset - contains labels of a single disease
    """
    for label, column in zip(disease_vector_labels, df_disease_columns):
        print(label, column)
        selected_diseases_df = df[[column]]
        selected_diseases_list = [label]
        desired_indices_list = find_desired_indices(labels, selected_diseases_list)
        selected_labels = selected_diseases_df.values
        X_subset = create_data_subset(desired_indices_list, X)
        y_subset = create_subset_labels(desired_indices_list, selected_labels)

    return X_subset, y_subset

##################################
# Data Augmentation
def augment_one_class(X, augmentation_method, num_total_images=5000):
    quotient = math.floor(abs((num_total_images - X.shape[0])) / X.shape[0])
    remainder = abs((num_total_images - X.shape[0])) % X.shape[0]

    final_augmented_list = []

    # if quotient == 0, this loop will not run
    for loop in range(quotient):
        augmented_images = augmentation_method(images=X)
        final_augmented_list.append(augmented_images)

    # Convert list to array
    final_augmented_array = np.array(final_augmented_list)

    # re-shape array from (#loops, disease count, img size, img size, channels) to 
    # (loops * disease count, img size, img size, channels))
    final_augmented_array = final_augmented_array.reshape(quotient * X.shape[0], \
        X.shape[1],  X.shape[2], X.shape[3])

    # Convert to list since appending array takes up a ton of memory
    final_augmented_list = list(final_augmented_array)

    # delete array since it takes up memory
    del final_augmented_array

    # Make list of random indices, length of the remainder
    rand_idxs = random.sample(range(X.shape[0]), remainder)

    for rand_idx in rand_idxs:
        augmented_images = augmentation_method(image=X[rand_idx])
        final_augmented_list.append(augmented_images)
    
    # add original images
    for original in list(X):
        final_augmented_list.append(original)

    # convert to array
    final_augmented_array = np.array(final_augmented_list)

    return final_augmented_array

