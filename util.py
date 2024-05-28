"""
    Author(s): Nathaniel del Rosario
"""

# import essentials
import scipy
import pandas as pd
import numpy as np

# methods & metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.datasets import make_blobs

# util
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import glob
import warnings
import os

def dataloader(filepath = 'data/Non_Demented'):
    """
    Loads the data for a specified directory

    Parameters
    ----------
    filepath : str
        directory for the images
    
    Returns
    -------
    images : numpy array
        the images converted to an array
    """
    image_files = glob.glob(os.path.join(filepath, '*.jpg'))

    images = []
    for image_file in image_files:
        # Open the image file
        img = Image.open(image_file)
        # Convert the image to a NumPy array
        img_array = np.array(img)
        images.append(img_array)

    images = np.array(images)
    print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')
    return images