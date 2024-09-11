import os
import cv2
import torch

# Create the output folder if it doesn't exist
output_folder = "annotated_frames"
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv5 pre-trained model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can replace 'yolov5s' with any other YOLO model version

# Set model confidence threshold if needed (default is 0.25)
model.conf = 0.4  # Optional: Adjust this based on your requirements

# Process frames from video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Run object detection on the frame
        results = model(frame)

        # Annotate the frame with the detection results
        results.render()  # This modifies the frame in-place

        # Save the annotated frame to the output folder
        annotated_frame_path = os.path.join(output_folder, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(annotated_frame_path, results.imgs[0])  # Save the frame with annotations

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames...")

    cap.release()
    print(f"Finished processing {frame_number} frames from video.")


# Process frames from an image folder
def process_image_folder(input_folder):
    images = os.listdir(input_folder)
    images.sort()  # Sort the images to maintain frame sequence

    frame_number = 0
    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Could not load image {image_name}. Skipping...")
            continue

        # Run object detection on the frame
        results = model(frame)

        # Annotate the frame with the detection results
        results.render()  # This modifies the frame in-place

        # Save the annotated frame to the output folder
        annotated_frame_path = os.path.join(output_folder, f"annotated_{frame_number:05d}.jpg")
        cv2.imwrite(annotated_frame_path, results.imgs[0])  # Save the frame with annotations

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames...")

    print(f"Finished processing {frame_number} frames from the folder.")


# Automate the process by detecting if input is a video file or a folder
def automate(input_path):
    if os.path.isfile(input_path):  # It's a video file
        print(f"Processing video: {input_path}")
        process_video(input_path)
    elif os.path.isdir(input_path):  # It's a folder with images
        print(f"Processing image folder: {input_path}")
        process_image_folder(input_path)
    else:
        print("Error: Invalid input path. Please provide a valid video file or folder path.")


# Example of how to call the function (Replace with your actual path)
input_path = "path_to_your_video_or_image_folder"  # Specify the path to your video file or image folder
automate(input_path)
