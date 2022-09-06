"""
Checks our format for our image and mask dirs.

We assume the dirs are formatted as follows
image-or-mask-dir
    label-name
        img1
        img2..
    label-name2...

    default extensions we use

    default format we use '_' + '[0-9]+'

Arguments:

    -h, --help            show this help message and exit
    -d, --ImageMaskDir
        image or mask dir where our data is stored. expected format shown above
    optional arguments
    -f --format
        format for the image/mask files. default is '_' + '[0-9]+'
    -e --extensions
        extensions we run our search on must be a list. default is ['jpg', 'tiff', 'png', 'jpeg']
    -o --outputPath
        save path for our checked dict
"""

import os
import glob
import re
from pathlib import Path
import argparse
import pickle


#takes in a label dir where the basename is the class
def check_image_mask_dir_label_file_format(label_dir, image_extensions, format):
    label = os.path.basename(label_dir)
    file_list = []
    for ext in image_extensions:
        glob_path = os.path.join(label_dir, '*.' + ext)
        file_list.extend(glob.glob(glob_path))
    my_format = label + format
    rex = re.compile(my_format)
    correct_format = []
    incorrect_format = []

    for file_path in file_list:

        file = Path(os.path.basename(file_path)).stem
        match = re.findall(rex, file)
        if match:
            if file in correct_format:
                print(f'Duplicate file found.\nBase name: {file}\n'
                      f'Full path {file_path}')
            else:
                correct_format.append(file)

        else:
            incorrect_format.append(file)
    checked = {
        'label_dir': label_dir,
        'incorrect_format': incorrect_format,
        'correct_format': correct_format,
        'ext': image_extensions
    }
    return checked


def check_image_mask_dir_file_format(image_mask_dir,
                                     image_extensions=['jpg', 'tiff', 'png', 'jpeg'], format='_' + '[0-9]+' ):
    #was getting dupliactes of jpeg and JPEG so took this out for now
    #image_extensions.extend([ext.upper() for ext in image_extensions if ext.upper() not in image_extensions])

    label_names = next(os.walk(image_mask_dir))[1]

    checked_dict = {}

    for label in label_names:
        label_dir = os.path.join(image_mask_dir, label)
        checked_dict[label] = check_image_mask_dir_label_file_format(label_dir, image_extensions, format)
    return checked_dict


def save_checked_dict(checked_dict, save_path):
    with open(save_path, 'wb') as fp:
        pickle.dump(checked_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def main():

    parser = argparse.ArgumentParser(
        description="Checking format for image/mask files. Defined above")

    parser.add_argument("-d",
                        "--ImageMaskDir",
                        help="directory where our image or mask files are stored",
                        type=str)
    parser.add_argument("-f",
                        "--format",
                        default='_' + '[0-9]+',
                        help="regex format were looking to match our image files to",
                        type=str)
    parser.add_argument("-e",
                        "--extensions",
                        help="extensions used for searching image/mask dirs",
                        default= ['jpg', 'tiff', 'png', 'jpeg'],
                        nargs="*",
                        type=str)
    parser.add_argument("-o",
                        "--outputPath",
                        help="output save path for our checked dict",
                        type=str)

    args = parser.parse_args()

    checked_dict = check_image_mask_dir_file_format(image_mask_dir=args.ImageMaskDir,
                                     format=args.format,
                                     image_extensions=args.extensions)
    save_checked_dict(checked_dict, args.outputPath)


if __name__ == '__main__':
    main()
