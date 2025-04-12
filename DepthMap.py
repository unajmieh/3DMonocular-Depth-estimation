import cv2  
import numpy as np  
import os  

def process_depth_maps(image_paths, output_directory):  
    # Check if output directory exists, if not, create it  
    if not os.path.exists(output_directory):  
        os.makedirs(output_directory)  

    for image_path in image_paths:  
        # Load the depth map image as grayscale  
        depth_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

        # Check if the image was loaded properly  
        if depth_map is None:  
            print(f"Error: Image {image_path} not found.")  
            continue  

        # 1. Contrast Adjustment using histogram equalization  
        equalized_depth_map = cv2.equalizeHist(depth_map)  

        # 2. Filtering using Gaussian blur  
        blurred_depth_map = cv2.GaussianBlur(equalized_depth_map, (5, 5), 0)  

        # Display the results (optional, can be commented out)  
        cv2.imshow('Original Depth Map', depth_map)  
        cv2.imshow('Equalized Depth Map', equalized_depth_map)  
        cv2.imshow('Blurred Depth Map', blurred_depth_map)  

        # Wait for a short time to allow displaying the images  
        cv2.waitKey(1)  # Change to 0 for blocking until key press, 1 for brief display  

        # Create unique filenames for the processed images  
        base_name = os.path.basename(image_path)  
        equalized_output_filename = os.path.join(output_directory, f'equalized_{base_name}')  
        blurred_output_filename = os.path.join(output_directory, f'blurred_{base_name}')  

        # Save the processed images  
        cv2.imwrite(equalized_output_filename, equalized_depth_map)  
        cv2.imwrite(blurred_output_filename, blurred_depth_map)  
        print(f"Processed and saved: {equalized_output_filename} and {blurred_output_filename}")  

    cv2.destroyAllWindows()  # Close all OpenCV windows when done  

# Example usage  
# Dynamically list all color-mapped images in the output directory  
depth_map_list = [os.path.join('./output_colors', file_name) for file_name in os.listdir('./output_colors') if file_name.startswith('color_mapped_') and file_name.endswith('.png') or file_name.endswith('.jpg')]  
output_dir = './processed_depth_maps'  # Directory to save processed images  

process_depth_maps(depth_map_list, output_dir)