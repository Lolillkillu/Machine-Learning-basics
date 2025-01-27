import cv2
import os
import numpy as np

#idk czy nie bedzie trzeb zmienić pryjmowanych argumentów w funkcji po dopisaniu nowych 
def resize_image(image_path, output_path, size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error {image_path}")

    resized_image = cv2.resize(image, size)

    cv2.imwrite(output_path, resized_image)

def augment_image(image):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)  # horyzontalne
    # idk czy nie bedzie trzeba dodać wiecej augmentacji
    return image

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def normalize_image(image):
    return image / 255.0

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def process_images(image_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error {image_path}")
            continue
        
        #użycie funkcji idk jeszcze jak to z modelem połączyć 
    
