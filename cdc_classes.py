# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:02:25 2023

@author: Setareh
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import  img_as_ubyte
from cdc_utils import (preprocessing, crop_image,
                             get_droplet_region, crop_by_mask)


 

class SimpleBlobDetection:
    
    def __init__(self, images, image_num, row, col, mask_droplet, background_correction):
        self.images = images
        self.image_num= image_num
        self.row = row
        self.col = col
        self.mask_droplet = mask_droplet
        self.background_correction = background_correction
    
    
    def simple_blob_detection_method(self):
    
        preprocessed_img = preprocessing(self.images[self.image_num,], self.background_correction )
        # Converts the image to an 8-bit unsigned integer format (0, 255)
        image = img_as_ubyte(preprocessed_img)
        # Create a single droplet from the image at (row, col) 
        droplet = crop_image(image, self.row, self.col, num_rows=9, num_columns=4)
       
        if self.mask_droplet == True:
            # Get the droplet region and the original grayscale image
            mask, image_gray = get_droplet_region(droplet)
            
            # Crop the original image using the droplet mask
            masked_droplet = crop_by_mask(mask, image_gray)
        else:
            masked_droplet = droplet
        
        keypoints = self.set_params(masked_droplet)
        classifications = self.classify_blobs(masked_droplet, keypoints)
        
        image_with_keypoints = cv2.drawKeypoints(masked_droplet, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(12,6))
        plt.imshow(image_with_keypoints)
        # plt.title('Detected Blobs'), plt.axis('off')
        if classifications:
            plt.title(', '.join(classifications)), plt.axis('off')
        else:
            plt.title('No Blob Detected.'), plt.axis('off')
    
    
        
        print("Blob Classifications based on Intensity:", classifications)
        
        return image_with_keypoints, classifications
    
    
    
    def set_params(self, cropped_droplet):
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Activates the area filter in the SimpleBlobDetector parameters
        params.filterByArea = True
        params.minArea = 25
        
        # Activates the circularity filter 
        params.filterByCircularity = True
        params.minCircularity = 0.2
        
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Specifies how elongated a shape is
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cropped_droplet)
        
        return keypoints
    
    
    
    
    def classify_blobs(self, cropped_droplet, keypoints):
        
        average_intensities = []
        classifications = []
        
        for keypoint in keypoints:
            # Coordinates of the blob centre
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            # diameter = keypoint.size
            radius = int(keypoint.size // 2)
            x1, y1 = max(x - radius, 0), max(y - radius, 0)
            x2, y2 = min(x + radius, cropped_droplet.shape[1]), min(y + radius, cropped_droplet.shape[0])
            
            blob_roi = cropped_droplet[y1:y2, x1:x2]
            avg_intensity = np.mean(blob_roi)
            average_intensities.append(avg_intensity)
            classifications.append(f'Living Cell at [({x1},{y1}),({x2},{y2})]' if avg_intensity > 128 else f'Synthetic Bead  at [({x1},{y1}),({x2},{y2})]') 
    
        
        return classifications
    
    
    
    



class MatchTemplte:
    
    def __init__(self, templates, mode):
        self.templates = templates
        self.mode = mode
        
    def match_template(self, image, threshold=0.8):
        """
        Perform template matching with single/multiple template(s).
        
        Parameters:
        - image: Grayscale image where objects are to be detected.
        - templates: List of grayscale template images.
        - threshold: Matching threshold (0 to 1).
        
        Returns:
        - result_image: Image with detected objects highlighted.
        """
        # Converts the image to an 8-bit unsigned integer format (0, 255)
        image = img_as_ubyte(image)
        result_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        result_images_list = []
        bounding_boxes = []
        classifications = []  
                
        for template in self.templates:
            
            image = img_as_ubyte(image)
            # Convert the grayscale image to BGR format for coloured rectangles/visualisation
            result_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            # Perform template matching
            template = img_as_ubyte(template)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where the matching score is above the threshold
            if self.mode=='single':
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(result_image,top_left, bottom_right, 255, 2)
                bounding_boxes.append([(top_left[0], top_left[1], w, h)])
    
    
                x1, y1 = top_left[0], top_left[1]
                x2, y2 = top_left[0] + w, top_left[1] + h
          
                blob_roi = image[y1:y2, x1:x2]
                avg_intensity = np.mean(blob_roi)
                classifications.append(f'Living Cell at [({x1},{y1}),({x2},{y2})]' if avg_intensity > 128 else f'Synthetic Bead  at [({x1},{y1}),({x2},{y2})]') 

                
                
            elif self.mode=='multi':
                loc = np.where(res >= threshold)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                bounding_box_multi = []
                # Draw rectangles around matched regions
                for pt in zip(*loc[::-1]):             
                    cv2.rectangle(result_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                    bounding_box_multi.append([(pt[0], pt[1], w, h)])
                    
                bounding_boxes.append(bounding_box_multi)
            else:
                print('Please enter a valid mode [single/multi].')
           
            result_images_list.append(result_image)
                
                
        return result_images_list, bounding_boxes, classifications
    
    
    
    
    def plot_detected_blobs(self, images, image_number, row, col, class_name, mask_droplet, background_correction):
        
        image = img_as_ubyte(images[image_number,])
        preprocessed_img = preprocessing(image, background_correction) 
        
        droplet = crop_image(preprocessed_img, row, col, num_rows=9, num_columns=4)
       
        if mask_droplet ==  True:
            # Get the droplet region and the original grayscale image
            mask, image_gray = get_droplet_region(droplet)
            
            # Crop the original image using the droplet mask
            masked_droplet = crop_by_mask(mask, image_gray)
        else:
            
            masked_droplet = droplet
            
        detected_images, bounding_boxes, classifications = self.match_template(masked_droplet, threshold=0.8)
       
        for n in range(len(self.templates)): 
            
            plt.figure(figsize=(12,6))
            plt.subplot(131)
            plt.imshow(self.templates[n], cmap=plt.cm.gray)
            plt.title(f'Template ({n})'), plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(droplet, cmap=plt.cm.gray)
            plt.title('Droplet'), plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(detected_images[n], cmap=plt.cm.gray)
            # plt.title('Detected '+ class_name + ' cell'), plt.axis('off')
            plt.title(f'Detected {classifications[n]} cell'), plt.axis('off')

        return bounding_boxes






# 



