import numpy as np
import pandas as pd
import cv2

class DatasetPreprocessing:
    @staticmethod
    def preprocess_dataset(dataset):
        # Preprocess and transform the dataset
        # e.g., resizing, normalizing, augmenting images
        
        # Example code for resizing images
        resized_images = []
        for image in dataset:
            resized_image = cv2.resize(image, (224, 224))
            resized_images.append(resized_image)
        
        return resized_images
    
    @staticmethod
    def split_dataset(dataset):
        # Split the dataset into training, validation, and testing sets
        
        # Example code for splitting the dataset
        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - val_size
        
        train_set = dataset[:train_size]
        val_set = dataset[train_size:train_size+val_size]
        test_set = dataset[train_size+val_size:]
        
        return train_set, val_set, test_set