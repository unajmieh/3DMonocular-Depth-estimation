import os
import subprocess
import json
import requests
from ultralytics import YOLO  # Example with YOLOv8
from label_studio_sdk import Client

# Step 1: Install and Start Label Studio
def start_label_studio():
    try:
        subprocess.run(["pip", "install", "label-studio"], check=True)
        subprocess.Popen(["label-studio"], stdout=subprocess.PIPE)
        print("Label Studio started at http://localhost:8080")
    except Exception as e:
        print(f"Error starting Label Studio: {e}")

# Step 2: Set Up Label Studio Project
def setup_label_studio_project(api_url, api_key, project_name, label_config):
    client = Client(url=api_url, api_key=api_key)
    project = client.start_project(
        title=project_name,
        label_config=label_config,
    )
    return project

# Step 3: Run Pre-Trained Model (YOLO) for Auto-Annotation
def generate_annotations_with_yolo(image_dir, output_dir):
    model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model
    results = model.predict(source=image_dir, save=True)
    
    annotations = []
    for result in results:
        for box in result.boxes.xyxy:  # Bounding box coordinates
            annotations.append({
                "image": os.path.basename(result.path),
                "bbox": box.tolist(),
                "label": result.names[int(box.cls)]
            })
    
    os.makedirs(output_dir, exist_ok=True)
    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f)
    
    print(f"Annotations saved to {annotations_path}")
    return annotations_path

# Step 4: Import Annotations into Label Studio
def import_annotations_to_label_studio(api_url, api_key, project_id, image_dir, annotations_path):
    client = Client(url=api_url, api_key=api_key)
    project = client.get_project(project_id)

    # Upload images to the project
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                project.upload_data(file=f)

    # Import annotations
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    tasks = []
    for annotation in annotations:
        tasks.append({
            "data": {"image": annotation["image"]},
            "annotations": [{
                "result": [{
                    "value": {
                        "x": annotation["bbox"][0],
                        "y": annotation["bbox"][1],
                        "width": annotation["bbox"][2] - annotation["bbox"][0],
                        "height": annotation["bbox"][3] - annotation["bbox"][1],
                    },
                    "type": "rectanglelabels",
                    "to_name": "image",
                    "from_name": "label",
                    "labels": [annotation["label"]],
                }]
            }]
        })
    
    project.import_tasks(tasks)
    print("Annotations imported successfully.")

# Step 5: Define Label Configuration (XML)
LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Car" background="blue"/>
    <Label value="Person" background="green"/>
  </RectangleLabels>
</View>
"""

# Main Function to Run the Entire Pipeline
def main():
    # Variables
    api_url = "http://localhost:8080"
    api_key = "<YOUR_LABEL_STUDIO_API_KEY>"  # Optional if API key is set up in Label Studio
    project_name = "Auto-Labeling Project"
    image_dir = "/path/to/your/images"
    output_dir = "/path/to/output/annotations"

    # Start Label Studio
    start_label_studio()

    # Set up Label Studio Project
    print("Setting up Label Studio project...")
    project = setup_label_studio_project(api_url, api_key, project_name, LABEL_CONFIG)

    # Generate Annotations with YOLO
    print("Generating annotations with YOLO...")
    annotations_path = generate_annotations_with_yolo(image_dir, output_dir)

    # Import Annotations into Label Studio
    print("Importing annotations into Label Studio...")
    import_annotations_to_label_studio(api_url, api_key, project.id, image_dir, annotations_path)

if __name__ == "__main__":
    main()
