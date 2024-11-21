import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import conversions as utils
import argparse
import shutil
def parse_cvat_annotation(xml_file):
    """
    Parses a CVAT annotation XML file and returns a list of annotations and a mapping of labels.

    Parameters:
    - xml_file (str): The path to the XML file to parse.

    Returns:
    - annotations (list): A list of dictionaries representing the annotations. Each dictionary contains the following keys:
        - id (str): The ID of the image.
        - name (str): The name of the image.
        - width (int): The width of the image.
        - height (int): The height of the image.
        - masks (list): A list of dictionaries representing the masks in the image. Each dictionary contains the following keys:
            - label (str): The label of the mask.
            - source (str): The source of the mask.
            - occluded (int): The occlusion status of the mask (0 for not occluded, 1 for occluded).
            - rle (str): The run-length encoding of the mask.
            - left (int): The left coordinate of the mask.
            - top (int): The top coordinate of the mask.
            - width (int): The width of the mask.
            - height (int): The height of the mask.
            - z_order (int): The z-order of the mask.
        - labels (list): A list of labels in the image.
        - points (list): A list of dictionaries representing the points in the image. Each dictionary contains the following keys:
            - label (str): The label of the points.
            - source (str): The source of the points.
            - occluded (int): The occlusion status of the points (0 for not occluded, 1 for occluded).
            - points (list): A list of tuples representing the points.
            - z_order (int): The z-order of the points.

    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []
    labels_mapping = {}

    # Parse labels from the <labels> section
    for label_tag in root.findall(".//labels/label"):
        label_name = label_tag.find('name').text
        label_color = label_tag.find('color').text
        labels_mapping[label_name] = {'color': label_color}

    for image_tag in root.findall(".//image"):
        image_info = {
            'id': image_tag.get('id'),
            'name': image_tag.get('name'),
            'width': int(image_tag.get('width')),
            'height': int(image_tag.get('height')),
            'masks': [],
            'labels': [],
            'points': []
        }

        for mask_tag in image_tag.findall(".//mask"):
            mask_info = {
                'label': mask_tag.get('label'),
                'source': mask_tag.get('source'),
                'occluded': int(mask_tag.get('occluded')),
                'rle': mask_tag.get('rle'),
                'left': int(mask_tag.get('left')),
                'top': int(mask_tag.get('top')),
                'width': int(mask_tag.get('width')),
                'height': int(mask_tag.get('height')),
                'z_order': int(mask_tag.get('z_order'))
            }

            image_info['masks'].append(mask_info)
            image_info['labels'].append(mask_info['label'])

        for points_tag in image_tag.findall(".//points"):
            points_info = {
                'label': points_tag.get('label'),
                'source': points_tag.get('source'),
                'occluded': int(points_tag.get('occluded')),
                'points': [tuple(map(float, point.split(','))) for point in points_tag.get('points').split(';')],
                'z_order': int(points_tag.get('z_order'))
            }

            image_info['points'].append(points_info)

        annotations.append(image_info)

    return annotations, labels_mapping

def copy_images(source_dir, destination_dir):
    """
    Copy images from the source directory to the destination directory.

    Args:
        source_dir (str): The path to the source directory containing the images.
        destination_dir (str): The path to the destination directory where the images will be copied to.

    Returns:
        None

    Raises:
        OSError: If the destination directory cannot be created.

    Example:
        copy_images('path/to/source', 'path/to/destination')
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    ann_directory = destination_dir.replace('images', 'cls_masks')
    ann_files = os.listdir(ann_directory)
    for root, dirs, files in os.walk(os.path.join(source_dir, "images")):
        for file in files:
            
            filename, ext = os.path.splitext(file)
            if ext.lower() not in ['.jpg', '.png', '.jpeg', '.gif']:
                continue
            if file in os.listdir(destination_dir):
                continue
            if file in ann_files:
                print(root)
                shutil.copy2(os.path.join(root, file), destination_dir)

    print("Images copied successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVAT Annotation Conertor to stemseg Annotation')
    parser.add_argument('--xml', type=str, help='Path to the CVAT XML file')
    parser.add_argument('--dest_path', type=str, help='Path to the CVAT annoations storage path')
    parser.add_argument('--save_images', action='store_true', help='Whether to save the images')
    parser.add_argument('--save_stems', action='store_true', help='Whether to save the images')
    
    args = parser.parse_args()
    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    annotation_data, labels_mapping = parse_cvat_annotation(args.xml)

    class_directory = os.path.join(args.dest_path, "cls_masks")
    if not os.path.exists(class_directory):
        os.makedirs(class_directory, exist_ok=True)
    stem_directory = os.path.join(args.dest_path, "stem_masks")
    if not os.path.exists(stem_directory):
        os.makedirs(stem_directory, exist_ok=True)

    for image_info in annotation_data:
        if f'{image_info["name"]}' in os.listdir(class_directory) and f'{image_info["name"]}' in os.listdir(stem_directory):
            continue
        print(f"Image ID: {image_info['id']}, Name: {image_info['name']}")
        print(f"Image Size: {image_info['width']} x {image_info['height']}")
        width, height = image_info['width'], image_info['height']
        mask_class = np.zeros((height, width, 3), dtype=np.uint8)

        for mask_info in image_info['masks']:
            try:
                cvat_rle = utils.deserialize_cvat_rle(mask_info)
            except:
                continue
            mask = utils.cvat_rle_to_binary_image_mask(cvat_rle, image_info['height'], image_info['width'])
            mask_class[mask>0] = tuple(int(labels_mapping[mask_info['label']]['color'][i:i + 2], 16) for i in (1, 3, 5))

        mask_class = Image.fromarray(mask_class)
        
       
        mask_class.save(f'{class_directory}/{image_info["name"]}')
            
        # Draw points on the mask_stem with radius 5 using label_mapping



        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for points_info in image_info['points']:
            print(f"  Points Label: {points_info['label']}")
            print(f"    Source: {points_info['source']}")
            print(f"    Occluded: {points_info['occluded']}")
            print(f"    Points: {points_info['points']}")
            print(f"    Z-Order: {points_info['z_order']}")

            # Get label color
            label_color = labels_mapping[points_info['label']]['color']
            color = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            # Convert points to integers
            points = [(int(x), int(y)) for (x, y) in points_info['points']]

            # Draw circular masks around key points
            for point in points:
                mask = utils.create_circle_mask((height, width), point, radius=10)
                mask_rgb[mask > 0] = color

        mask_rgb = Image.fromarray(mask_rgb)
        
        
        mask_rgb.save(f'{stem_directory}/{image_info["name"]}')

    
    if args.save_images:
        # get directory name two levels down
        source_dir = os.path.dirname(args.xml)
        print(source_dir)
        dest_dir = os.path.join(args.dest_path, "rgb")
        shutil.move(os.path.join(source_dir, "images"), dest_dir)
    if not args.save_stems:
        shutil.rmtree(stem_directory)
                