import os  
import cv2  
import numpy as np  
import requests  

def process_image(image_path):  
    # Check if the image is a URL or a local file  
    if image_path.startswith('http://') or image_path.startswith('https://'):  
        response = requests.get(image_path)  
        
        # Check if the request was successful  
        if response.status_code != 200:  
            print(f"Failed to retrieve the image from URL: {image_path}")  
            return  
        image_np = np.frombuffer(response.content, np.uint8)  
        image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)  
    else:  
        image = cv2.imread(image_path)  

    # Check if the image was loaded  
    if image is None:  
        print(f"Failed to load the image: {image_path}")  
        return  

    # Convert image to RGB  
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Create a mask for white areas  
    lower_white = np.array([200, 200, 200])  # Lower bound for white  
    upper_white = np.array([255, 255, 255])  # Upper bound for white  
    white_mask = cv2.inRange(rgb_image, lower_white, upper_white)  

    # Create masks for red and green areas  
    lower_red = np.array([150, 0, 0])  # Lower bound for red  
    upper_red = np.array([255, 100, 100])  # Upper bound for red  
    red_mask = cv2.inRange(rgb_image, lower_red, upper_red)  
    
    lower_green = np.array([0, 100, 0])  # Lower bound for green  
    upper_green = np.array([100, 255, 100])  # Upper bound for green  
    green_mask = cv2.inRange(rgb_image, lower_green, upper_green)  

    # Prepare an output image where white areas will turn to blue  
    output_image = rgb_image.copy()  

    # Change white areas to blue  
    output_image[white_mask > 0] = [0, 0, 255]  # Change white pixels to blue  

    # Preserve red and green areas  
    output_image[red_mask > 0] = rgb_image[red_mask > 0]  
    output_image[green_mask > 0] = rgb_image[green_mask > 0]  

    # Create a unique filename for the modified image  
    base_name = os.path.basename(image_path)  
    output_filename = f'modified_{base_name}'  
    cv2.imwrite(output_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  
    print(f"Modified image saved as {output_filename} successfully.")  

# Example usage processing all images in the output_colors directory  
def process_images_in_directory(directory):  
    # List all files in the specified directory  
    for file_name in os.listdir(directory):  
        file_path = os.path.join(directory, file_name)  
        # Check if the file is an image file  
        if file_name.endswith('.png') or file_name.endswith('.jpg'):  
            process_image(file_path)  

# Run the function on the output directory  
output_colors_directory = './output_colors'  
process_images_in_directory(output_colors_directory)