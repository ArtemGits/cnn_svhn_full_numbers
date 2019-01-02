import os
import sys
import json
import h5py
import tarfile
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from unpacker import DigitStructWrapper
from scipy.ndimage import imread
from scipy.misc import imresize

#TODO create class for working with tar archives
def extract_tarball(filename, force=False):
    """ Helper function for extracting tar archive file """
    # Drop the file extension
    root = filename.split('.')[0] 
    
    # If file is already extracted - return
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
        return
    
    # If file is a tarball file - extract it
    if (filename.endswith("tar.gz")):
        print("Extracting %s ..." % filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()


#TODO create class for getting bounding boxes
def get_bounding_boxes(start_path = '.'):
    """ Extracts a bounding box file and returns a dictionary"""
    return DigitStructWrapper(start_path).unpack_all()

#TODO create class for work with dataframe
def dict_to_dataframe(image_bounding_boxes, path):
    """ Helper function for flattening the bounding box dictionary
    """
    # Store each bounding box
    boxes = []
    
    # For each set of bounding boxes
    for image in tqdm(image_bounding_boxes):
        
        # For every bounding box
        for bbox in image['boxes']:
            
            # Store a dict with the file and bounding box info
            boxes.append({
                    'filename': path + image['filename'],
                    'label': bbox['label'],
                    'width': bbox['width'],
                    'height': bbox['height'],
                    'top': bbox['top'],
                    'left': bbox['left']})
            
    # return the data as a DataFrame
    return pd.DataFrame(boxes)

#TODO create class for utils functions
def get_image_size(filepath):
    """Returns the image size in pixels given as a 2-tuple (width, height)
    """
    image = Image.open(filepath)
    return image.size 

def get_image_sizes(folder):
    """Returns a DataFrame with the file name and size of all images contained in a folder
    """
    image_sizes = []
    
    # Get all .png images contained in the folder
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    
    # Get image size of every individual image
    for i,image in enumerate(images):
        w, h = get_image_size(folder + image)
        image_size = {'filename': folder + image, 'image_width': w, 'image_height': h}
        image_sizes.append(image_size)
        if i%10000==0: 
            print(i)
        
    # Return results as a pandas DataFrame
    return pd.DataFrame(image_sizes)


def crop_and_resize(image, img_size):
    """ Crop and resize an image
    """
    image_data = imread(image['filename'])
    crop = image_data[image['y0']:image['y1'], image['x0']:image['x1'], :]
    return imresize(crop, img_size)


def create_dataset(df, img_size):
    """ Helper function for converting images into a numpy array
    """
    # Initialize the numpy arrays (0's are stored as 10's)
    X = np.zeros(shape=(df.shape[0], img_size[0], img_size[0], 3), dtype='uint8')
    y = np.full((df.shape[0], 5), 10, dtype=int)
    
    # Iterate over all images in the pandas dataframe (slow!)
    for i, (index, image) in enumerate(df.iterrows()):
        if i%10000==0:
            print(i)
        # Get the image data
        X[i] = crop_and_resize(image, img_size)
        
        # Get the label list as an array
        labels = np.array((image['labels']))
                
        # Store 0's as 0 (not 10)
        labels[labels==10] = 0
        
        # Embed labels into label array
        y[i,0:labels.shape[0]] = labels
        
    # Return data and labels   
    return X, y

def random_sample(N, K):
    """Return a boolean mask of size N with K selections
    """
    mask = np.array([True]*K + [False]*(N-K))
    np.random.shuffle(mask)
    return mask


def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    greyscale = np.dot(images, [0.2989, 0.5870, 0.1140])
    return np.expand_dims(greyscale, axis=3)


def my_agg(x):
    names = {
            'x0': x['x0'].min(),
            'y0': x['y0'].min(),
            'x1': x['x1'].max(),
            'y1': x['y1'].max(),
            'labels': list(x['label']),   #TODO test x variable scope
            'num_digits': x['label'].count()
             }
    return pd.Series(names, index=['x0', 'y0', 'x1','y1', 'labels', 'num_digits'])


def main():
    # os.listdir("data") -----
    # Get the directory listing for the dataset folder
    ls_data = [f for f in os.listdir("data") if 'tar.gz' in f]  
    # cd data
    os.chdir("data")
    # Extract the tarballs
    extract_tarball(ls_data[0])
    extract_tarball(ls_data[1])
    #extract_tarball(ls_data[2])
    # cd ..
    os.chdir(os.path.pardir)

    # Extract the bounding boxes (this will take a while!)
    train_bbox = get_bounding_boxes('data/train/digitStruct.mat')
    print("Done ..........geting Train images bounding box")
    test_bbox = get_bounding_boxes('data/test/digitStruct.mat')
    print("Done ..........geting Test images bounding box")
    #extra_bbox = get_bounding_boxes('data/extra/digitStruct.mat')
    #print("Done ..........geting Extra images bounding box")
    
    # Display the information stored about an individual image
    print(json.dumps(train_bbox[0], indent=2))


    # We store the bounding boxes here
    bbox_file = 'data/bounding_boxes.csv'

    if not os.path.isfile(bbox_file):
    
        # Extract every individual bounding box as DataFrame  
        train_df = dict_to_dataframe(train_bbox, 'data/train/')
        test_df = dict_to_dataframe(test_bbox, 'data/test/')
        #extra_df = dict_to_dataframe(extra_bbox, 'data/extra/')

        print("Training", train_df.shape)
        print("Test", test_df.shape)
        #print("Extra", extra_df.shape)
        print('')

        # Concatenate all the information in a single file
        df = pd.concat([train_df, test_df]) #deleted extra_df
    
        print("Combined", df.shape)

        # Write dataframe to csv
        df.to_csv(bbox_file, index=False)

        # Delete the old dataframes
        del train_df, test_df, train_bbox, test_bbox 
    
    else:
        # Load preprocessed bounding boxes
        df = pd.read_csv(bbox_file)


    # Rename the columns to more suitable names
    df.rename(columns={'left': 'x0', 'top': 'y0'}, inplace=True)

    # Calculate x1 and y1
    df['x1'] = df['x0'] + df['width']
    df['y1'] = df['y0'] + df['height']

    # Apply the aggration
    df = df.groupby('filename').apply(my_agg).reset_index() #TODO WTF 
    
    # Fix the column names after aggregation
#    df.columns = [x[0] if i < 5 else x[1] for i, x in enumerate(df.columns.values)]
    
    # Display the results
    print(df.head())
    # Calculate the increase in both directions
    df['x_increase'] = ((df['x1'] - df['x0']) * 0.3) / 2.
    df['y_increase'] = ((df['y1'] - df['y0']) * 0.3) / 2.
    
    # Apply the increase in all four directions
    df['x0'] = (df['x0'] - df['x_increase']).astype('int')
    df['y0'] = (df['y0'] - df['y_increase']).astype('int')
    df['x1'] = (df['x1'] + df['x_increase']).astype('int')
    df['y1'] = (df['y1'] + df['y_increase']).astype('int')
    
    # Select the dataframe row corresponding to our image
    image = 'data/train/1.png'
    bbox = df[df.filename == image]


    # Extract the image sizes
    train_sizes = get_image_sizes('data/train/')
    print("Done............getting Train_Set sizes")
    test_sizes = get_image_sizes('data/test/')
    print("Done............getting Test_Set sizes")
    #extra_sizes = get_image_sizes('data/extra/')
    #print("Done............getting extra_Set sizes")
    
    # Concatenate all the information in a single file
    image_sizes = pd.concat([train_sizes, test_sizes])
    
    # Delete old dataframes
    del train_sizes, test_sizes 
    
    print("Bounding boxes", df.shape)
    print("Image sizes", image_sizes.shape)
    print('')
    
    # Inner join the datasets on filename
    df = pd.merge(df, image_sizes, on='filename', how='inner')
    
    print("Combined", df.shape)
    
    # Delete the image size df
    del image_sizes
    
    # Store checkpoint
    df.to_csv("data/image_data.csv", index=False)
    #df = pd.read_csv('data/image_data.csv')
    
    # Correct bounding boxes not contained by image
    df.loc[df['x0'] < 0, 'x0'] = 0
    df.loc[df['y0'] < 0, 'y0'] = 0
    df.loc[df['x1'] > df['image_width'], 'x1'] = df['image_width']
    df.loc[df['y1'] > df['image_height'], 'y1'] = df['image_width']

    print(df.head())
    
    # Count the number of images by number of digits
    df.num_digits.value_counts(sort=False)
    # Keep only images with less than 6 digits
    df = df[df.num_digits < 6]

    # Change this to select a different image size
    image_size = (32, 32)
    
    # Get cropped images and labels (this might take a while...)
    X_train, y_train = create_dataset(df[df.filename.str.contains('train')], image_size)
    print("Done Dataset ...............Train data")
    X_test, y_test = create_dataset(df[df.filename.str.contains('test')], image_size)
    print("Done Dataset ...............Test data")
    #X_extra, y_extra = create_dataset(df[df.filename.str.contains('extra')], image_size)
    #print("Done Dataset ...............extra data")
    # We no longer need the dataframe
    del df
    
    print("Training", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    #print('Extra', X_extra.shape, y_extra.shape)

    # Pick 4000 training and 2000 extra samples
    sample1 = random_sample(X_train.shape[0], 3000)
    sample2 = random_sample(X_test.shape[0], 2000)

    # Create valdidation from the sampled data
    X_val = np.concatenate([X_train[sample1], X_test[sample2]])
    y_val = np.concatenate([y_train[sample1], y_test[sample2]])

    # Keep the data not contained by sample
    X_train = np.concatenate([X_train[~sample1], X_test[~sample2]])
    y_train = np.concatenate([y_train[~sample1], y_test[~sample2]])


    print("Training", X_train.shape, y_train.shape)
    print('Validation', X_val.shape, y_val.shape)

    # Transform the images to greyscale
    X_train = rgb2gray(X_train).astype(np.float32)
    X_test = rgb2gray(X_test).astype(np.float32)
    X_val = rgb2gray(X_val).astype(np.float32)

    # Create file
    h5f = h5py.File('data/SVHN_multi_grey.h5', 'w')
    
    # Store the datasets
    h5f.create_dataset('train_dataset', data=X_train)
    h5f.create_dataset('train_labels', data=y_train)
    h5f.create_dataset('test_dataset', data=X_test)
    h5f.create_dataset('test_labels', data=y_test)
    h5f.create_dataset('valid_dataset', data=X_val)
    h5f.create_dataset('valid_labels', data=y_val)

    # Close the file
    h5f.close()
       
    
if __name__ == '__main__':
    main()

