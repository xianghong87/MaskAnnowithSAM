import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from utils import scan_all_files, get_boxes_from_voc
from options import parse_args


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def generate_color():
    color_list = []

    for i in range(1000):
        for i in range(3):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            rgb = [r, g, b]
        color_list.append(rgb)
    return color_list


if __name__ == '__main__':

    opt = parse_args()

    input_folder = opt.input
    xml_folder = opt.xml_folder
    output_folder = opt.output
    sam_checkpoint = opt.weight_path
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    color_list = generate_color()
    files = scan_all_files(input_folder, 'img')
    print(len(files))

    for file in files:
        print(file)
        img_path = file
        basename = os.path.splitext(os.path.basename(file))[0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xml_path = os.path.join(xml_folder, basename+'.xml')
        if os.path.exists(xml_path):
            classes = ['bone']
            box_list, _ = get_boxes_from_voc(xml_path, classes)

            input_boxes = torch.tensor(box_list, device=predictor.device)
            predictor.set_image(image)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            classMap = np.zeros(image.shape, dtype=np.uint8)
            grayMap = np.zeros(masks[0].cpu().numpy().squeeze().shape, dtype=np.uint8)
            for i in range(len(masks)):
                mask = masks[i].cpu().numpy().squeeze()
                classMap[mask] = color_list[i]
                grayMap[mask] = 255

            color = cv2.addWeighted(image, 0.6, classMap, 0.4, 10)

            output_name_color = os.path.join(output_folder, basename+'_color.png')
            output_name_gray = os.path.join(output_folder, basename + '_gray.png')

            cv2.imwrite(output_name_color, color)
            cv2.imwrite(output_name_gray, grayMap)
