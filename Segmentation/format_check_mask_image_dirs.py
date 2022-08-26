"""
Takes in a given format and checks a image and mask directory.

The format were using is {class-name}_{image/mask}_{file_id}.{ext}

for the extension if there are multiple put it in brackets e.g
[png][jpg][JPG]
for glob.glob '*' + '.[png][jpg][JPG]'
"""


import re
import os
import glob
import argparse

######
"""
Functions for checking image/mask dir
"""

"""
Sanity check functions, displaying args, and the incorrect, if any, formatted files
"""


def display_selected_args(image_or_mask, image_mask_dir, correct_format, total_paths):
    print(f'######\nChecking format for {total_paths} {image_or_mask}s files in {image_mask_dir}\n'
          f'Seleceted format were checking for {correct_format}\n#######')


def display_incorrect_formatted_paths(incorrect_formatted_paths, total_paths):
    num_incorrect_formatted_paths = len(incorrect_formatted_paths)

    if num_incorrect_formatted_paths > 0:
        print(f'Out of {total_paths} paths, {num_incorrect_formatted_paths} have the incorrect format.')
        y_n = True
        while y_n:
            u_input = input("Display incorrectly formatted paths?\n(y/n)")
            if u_input == 'y':
                for path in incorrect_formatted_paths:
                    print(f'\nBASENAME\n{os.path.basename(path)}')
                    print(f'\nFULL PATH\n{path}')
                y_n = False
            if u_input == 'n':
                y_n = False
    return


"""
File checking functions 
"""


def check_filename_format(rex, path):
    filename = os.path.basename(path)
    if rex.match(filename):
        return True
    else:
        return False


def check_format_of_paths(paths, correct_format):

    rex = re.compile(correct_format)
    correct_paths = []
    incorrect_paths = []

    for path in paths:
        if check_filename_format(rex, path):
            correct_paths.append(path)
        else:
            incorrect_paths.append(path)

    return correct_paths, incorrect_paths


def check_image_mask_dir(image_mask_dir, image_or_mask, class_name, ext, regex_file_id='[0-9]+'):

    correct_format = class_name + '_' + image_or_mask + '_' + regex_file_id + ext
    glob_path = os.path.join(image_mask_dir, '*' + ext)
    paths =  glob.glob(glob_path)
    total_paths = len(paths)
    display_selected_args(image_or_mask, image_mask_dir, correct_format, total_paths)

    correct_paths, incorrect_paths = check_format_of_paths(paths, correct_format)
    num_incorrect_paths = len(incorrect_paths)
    if num_incorrect_paths > 0:
        display_incorrect_formatted_paths(incorrect_paths, total_paths)
    else:
        print('All files formatted correctly')
######

######
"""
Check both image and masks in ...project-dir/Original-Images-Masks
Use this when there are no tiff files to check or just to check both 
"""


#layer of redundancy for checking if we have matching class names
def check_class_names(image_dir, mask_dir):
    image_class_names = next(os.walk(image_dir))[1]
    mask_class_names = next(os.walk(mask_dir))[1]

    assert len(image_class_names) == len(mask_class_names), \
        f'Different number of classes in Images {len(image_class_names)} than Masks {mask_class_names}'
    assert image_class_names == mask_class_names, \
        f'Found different class names in Images {image_class_names} than in Masks {mask_class_names}'
    print(f'Image and Mask class-name check complete\n'
          f'Found {image_class_names}')
    return image_class_names


def check_orginal_images_and_masks(orig_image_mask_dir, ext, regex_file_id='[0-9]+'):
    image_dir = os.path.join(orig_image_mask_dir, 'Images')
    mask_dir = os.path.join(orig_image_mask_dir, 'Masks')

    image_mask_dirs = [image_dir, mask_dir]
    class_names = check_class_names(image_dir, mask_dir)

    for image_mask_dir in image_mask_dirs:
        for name in class_names:
            image_or_mask = os.path.basename(image_mask_dir).lower()
            if image_or_mask == 'images':
                image_or_mask = 'image'
            if image_or_mask == 'masks':
                image_or_mask = 'mask'
            current_dir = os.path.join(image_mask_dir, name)
            check_image_mask_dir(current_dir, image_or_mask, name, ext, regex_file_id=regex_file_id)
    return


