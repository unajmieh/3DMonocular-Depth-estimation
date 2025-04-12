import cv2  
import numpy as np  
import os  

def color_map_images(image_paths, output_directory, color_ranges):  
    # Check if output directory exists, if not, create it  
    if not os.path.exists(output_directory):  
        os.makedirs(output_directory)  

    for image_path in image_paths:  
        image = cv2.imread(image_path)  
        # Check if image was loaded successfully  
        if image is None:  
            print(f"Error loading image: {image_path}")  
            continue  

        # Convert image to RGB  
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # Create an output image initialized to black  
        output_image = np.zeros_like(rgb_image)  
        # Apply color mapping  
        for lower, upper, color in color_ranges:  
            mask = cv2.inRange(rgb_image, lower, upper)  
            output_image[mask > 0] = color  

        # Create a unique filename based on the original image name  
        base_name = os.path.basename(image_path)  
        output_filename = os.path.join(output_directory, f'color_mapped_{base_name}')  
        # Save the modified image  
        cv2.imwrite(output_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  
        print(f"Color mapped image saved as {output_filename} successfully.")  

# Define your color ranges  
color_ranges = [  
    (np.array([0, 0, 100]), np.array([100, 100, 255]), [255, 0, 0]),  # Red  
    (np.array([0, 100, 0]), np.array([100, 255, 100]), [0, 255, 0]),  # Green  
    (np.array([100, 0, 0]), np.array([255, 100, 100]), [0, 0, 255]),  # Blue  
    (np.array([0, 0, 0]), np.array([100, 100, 100]), [255, 255, 0]),  # Yellow  
    # Add more ranges as needed  
]  

# Example usage  
image_list = [f'./Dataset/{file_name}' for file_name in os.listdir('./Dataset') if file_name.endswith('.png')]  
output_dir = './output_colors'  # Directory to save color mapped images  

color_map_images(image_list, output_dir, color_ranges)