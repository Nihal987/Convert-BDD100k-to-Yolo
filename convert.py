import json
import cv2
import os
import matplotlib.pyplot as plt

# Helper Functions

# Getting the image details from the dataset, by providing the filename of the image
def get_img_details(filename):
  for img in data['images']:
    if img['file_name'] == filename:
      return img

# This function takes an image_id as a parameter and returns the annotations of that image.
def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None

# Function to title all images in an incrementle way {img0,img1...}
def load_images_from_folder(folder):
    filenames = []
    count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            # os.chdir(output_path)
            cv2.imwrite(f"{output_path}/images/img{count}.jpg", img)
            filenames.append(filename)
            count += 1
    return filenames

# Setting Paths
image_input_path = "coco_val"
output_path = "/Users/nihal/Desktop/data/converted_coco_val"
annotations_input_path = "annotations/ins_seg_val_coco.json"

# Reading Annotations file
f = open(annotations_input_path)
data = json.load(f)
f.close()

# Processing Images
original_file_names = load_images_from_folder(image_input_path)


# Applying Conversion
count = 0
for filename in original_file_names:
  # Extracting image 
  img = get_img_details(filename)
  img_id = img['id']
  img_w = img['width']
  img_h = img['height']

  # Get Annotations for this image
  img_ann = get_img_ann(img_id)

  if img_ann:
    # Opening file for current image
    file_object = open(f"{output_path}/labels/img{count}.txt", "a")

    for ann in img_ann:
      if "bbox" not in ann.keys(): # In my dataset (bdd100k) background or crowd images don't have bbox
        continue
      current_category = ann['category_id'] - 1 # Subtracting 1 from category id because class labels in yolo format starts from 0.
      current_bbox = ann['bbox']
      x = current_bbox[0]
      y = current_bbox[1]
      w = current_bbox[2]
      h = current_bbox[3]
      
      # Finding midpoints
      x_centre = (x + (x+w))/2
      y_centre = (y + (y+h))/2
      
      # Normalization
      x_centre = x_centre / img_w
      y_centre = y_centre / img_h
      w = w / img_w
      h = h / img_h
      
      # Limiting upto fix number of decimal places
      x_centre = format(x_centre, '.6f')
      y_centre = format(y_centre, '.6f')
      w = format(w, '.6f')
      h = format(h, '.6f')
          
      # Writing current object 
      file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

    file_object.close()
  count += 1