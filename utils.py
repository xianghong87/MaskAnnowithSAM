import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def get_boxes_from_voc(xml_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    box_list = []
    classes_list = []
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]
        box_list.append(b)
        classes_list.append(cls_id)
    return box_list, classes_list


def scan_all_files(video_root, video_or_img='img'):
    # scan  multilevel directory
    # files_list = []
    file_list = []
    for root, sub_dirs, files in os.walk(video_root):
        for f_n in files:
            file_path = os.path.join(root, f_n)
            if is_video_or_img_file(file_path, video_or_img):
                file_list.append(file_path)
    return file_list


def is_video_or_img_file(filename, video_or_img='img'):
    if video_or_img == 'video':
        FILE_EXTENSIONS = ['.mp4', '.MP4']

    elif video_or_img == 'img':
        FILE_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.dcm',
            '.JPEG', '.png', '.PNG'
        ]

    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)


def normalise(img):

    max_value = np.max(img)
    min_value = np.min(img)

    out = (img - min_value)/(max_value - min_value)

    return out


def convert_dicon_to_uint8(dc):
    pixel_array_numpy = dc.pixel_array

    pixel_array_numpy_norm = normalise(pixel_array_numpy) * 255
    out_uint8 = cv2.convertScaleAbs(pixel_array_numpy_norm)

    out = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return out