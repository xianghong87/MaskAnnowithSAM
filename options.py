import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str,
                        default='data/images',
                        help='path to input folder')
    parser.add_argument('--xml_folder', type=str,
                        default='data/xmls',
                        help='path to labelImg folder')
    parser.add_argument('--output', type=str, default='data/out_images',
                        help='path to output folder')
    parser.add_argument('--weight_path', type=str,
                        default='checkpoints/sam_vit_h_4b8939.pth',
                        help='path to output folder')

    opt = parser.parse_args()
    print("input args:\n", json.dumps(vars(opt), indent=4, separators=(",", ":")))

    return opt