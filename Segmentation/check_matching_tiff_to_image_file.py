
""" usage:
check to make sure our mask and image files match
based off the number of the file we check if our tiff files have a corresponding image file and vice versa.

expected tiff format: {class-name}-{file_num}_{apeer_class_name}.ome.tiff
e.g: Cabernet-Sauvignon-0_Cabernet.ome.tiff
arguments:
    tiff dir
    image dir
    class name
    apeer class name
    ext
expected image file format: {class-name}_image_{file_num}.ext
e.g: Cabernet-Sauvignon_image_0.et
"""
import re
import glob
import os
import argparse


"""
checking format for tiff files
"""


def check_tiff_file_format(tiff_file, class_name, apeer_class_name):
    rex_pattern = "[0-9]+"
    extension = '.ome.tiff'
    dash_or_underscore = '[_\-]+'
    correct_format = class_name + dash_or_underscore + rex_pattern + dash_or_underscore + apeer_class_name + extension
    rex = re.compile(correct_format)
    if rex.match(tiff_file):
        return True
    else:
        return False


def get_valid_invalid_tiff_paths(tiff_mask_paths, class_name, apeer_class_name):
    valid_paths = []
    invalid_paths = []
    for tiff_path in tiff_mask_paths:
        tiff_file = os.path.basename(tiff_path)
        if check_tiff_file_format(tiff_file, class_name, apeer_class_name):
            valid_paths.append(tiff_path)
        else:
            invalid_paths.append(tiff_path)
    checked_tiff_paths = {
        'valid_paths': valid_paths,
        'invalid_paths': invalid_paths
    }
    return checked_tiff_paths


"""
checking if tiff files have matching image file 
"""

def get_digits_from_file(file):
    return (''.join([s for s in file if s.isdigit()]))


def make_image_filename_from_tiff_file(tiff_file, class_name, ext):
    file_num = get_digits_from_file(tiff_file)
    image_file = class_name + '_image_' + file_num + ext
    return image_file


#checking to see if tiff has a matching image
def get_valid_invalid_tiff_to_image_pairs(tiff_paths, image_dir, class_name, ext):

    valid_image_paths = []
    invalid_image_paths = []
    for tiff_path in tiff_paths:
        tiff_file = os.path.basename(tiff_path)
        image_filename = make_image_filename_from_tiff_file(tiff_file, class_name, ext)
        image_path = os.path.join(image_dir, image_filename)
        if os.path.exists(image_path):
            valid_image_paths.append(image_path)
        else:
            invalid_image_paths.append(tiff_path)
    checked_paths = {
        'valid_paths': valid_image_paths,
        'invalid_paths': invalid_image_paths
    }
    return checked_paths


#expected tiff format: {class-name}-{file_num}_{apeer_class_name}.ome.tiff
#Cabernet-Sauvignon-8_Cabernet.ome.tiff
#Cabernet-Sauvignon_13_Cabernet.ome.tiff
#returns the tiff format with the filenumber from image_file
def make_tiff_filename_from_image_file(image_file, class_name, apeer_class_name):
    file_num = get_digits_from_file(image_file)
    tiff_filename = class_name + "[-_]" + file_num + "[-_]" + apeer_class_name + '.ome.tiff'
    return tiff_filename


def get_valid_invalid_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name):
    valid_tiff_paths = []
    invalid_tiff_paths = []
    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        tiff_filename = make_tiff_filename_from_image_file(image_file, class_name, apeer_class_name)
        tiff_path = glob.glob(os.path.join(tiff_dir, tiff_filename))
        if len(tiff_path) > 0:
            valid_tiff_paths.append(tiff_path[0])
        elif len(tiff_path) > 1:
            print(f'Found {len(tiff_path)} matches for image file\n{image_file}')
        else:
            invalid_tiff_paths.append(os.path.join(tiff_dir, tiff_filename))
    checked_paths = {
        'valid_paths': valid_tiff_paths,
        'invalid_paths': invalid_tiff_paths
    }
    return checked_paths


def display_invalid_paths(invalid_paths):
    for path in invalid_paths:
        print('####')
        print('Following tiff file(s) have an invalid format')
        print(f'BASENAME\n{os.path.basename(path)}')
        print('####')
        print(f'FULL PATH\n{path}\n')


def display_total_paths(total_image_paths, total_mask_paths):
    if total_image_paths != total_mask_paths:
        print(f'Found different number of image and mask paths\n'
              f'Num image paths: {total_image_paths}\n'
              f'Num mask paths: {total_mask_paths}\n')
    else:
        print(f'####\nFound same number of image paths ({total_image_paths})\n'
              f'And tiff paths({total_mask_paths})')


def display_invalid_paths_pairs(invalid_paths):
    ui = input('Display invalid paths\n(y/n)')

    while True:
        if ui == 'y':
            for path in invalid_paths:
                tiff_file = os.path.basename(path)
                error_msg = f'\nCould not find matching pair for {tiff_file}\n'
                print(error_msg)
            return
        if ui =='n':
            return

def check_tiff_paths(tiff_mask_paths, class_name, apeer_class_name):
    checked_tiff_paths = get_valid_invalid_tiff_paths(tiff_mask_paths, class_name, apeer_class_name)

    invalid_tiff_paths = checked_tiff_paths['invalid_paths']
    num_invalid_tiff_paths = len(invalid_tiff_paths)
    if num_invalid_tiff_paths > 0:
        display_invalid_paths(invalid_tiff_paths)
    else:
        print('####\nAll tiff files formatted correctly!')


def remove_paths(invalid_paths):
    while True:
        ui = input(f'Remove {len(invalid_paths)} files from the directory: {os.path.dirname(invalid_paths[0])}?'
                   f'\n(y/n)')
        if ui == 'y':
            for path in invalid_paths:
                os.remove(path)
            return
        if ui == 'n':
            return


def check_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name):
    checked_image_to_tiff_pairs = \
        get_valid_invalid_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name)
    invalid_paths = checked_image_to_tiff_pairs['invalid_paths']
    print(f'####\nOut of {len(image_paths)} image paths, found {len(invalid_paths)} invalid pairs for image to tiff')
    if len(invalid_paths)>0:
        display_invalid_paths_pairs(invalid_paths)
        remove_paths(invalid_paths)


def check_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext):
    checked_tiff_to_image_pairs = get_valid_invalid_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext)
    invalid_paths = checked_tiff_to_image_pairs['invalid_paths']
    print(f'####\nOut of {len(tiff_mask_paths)} tiff paths, found {len(invalid_paths)} invalid pairs for tiff to image')
    if len(invalid_paths)>0:
        display_invalid_paths_pairs(invalid_paths)
        remove_paths(invalid_paths)


def check_tiff_path_format_and_tiff_image_pairs(tiff_dir, image_dir, class_name, apeer_class_name, ext='.jpg'):
    tiff_glob_path = os.path.join(tiff_dir, '*.tiff')
    tiff_mask_paths = glob.glob(tiff_glob_path)
    total_mask_paths = len(tiff_mask_paths)

    image_glob_path = os.path.join(image_dir,'*'+ext )
    image_paths = glob.glob(image_glob_path)
    total_image_paths = len(image_paths)

    display_total_paths(total_image_paths, total_mask_paths)

    check_tiff_paths(tiff_mask_paths, class_name, apeer_class_name)
    check_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext)

    check_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name)

    print('\nDone checking pairs!')


def main():
    parser = argparse.ArgumentParser(
        description="rename and saving mask files")

    parser.add_argument("-t",
                        "--TIFFDIR",
                        help="path to where the masks are stored as .tiff files",
                        type=str)
    parser.add_argument("-i",
                         "--IMAGEDIR",
                        help="where we save our newly made masks. default will be DIRS['MASK_DIR] + class_name",
                        type=str)
    parser.add_argument("-a",
                        "--APEERCLASSNAME",
                        type=str)
    parser.add_argument("-c",
                        "--CLASSNAME",
                        help="name of the class. defualt is name of the base dir for TIFFDIR",
                        nargs='?',
                        type=str)
    parser.add_argument("-e",
                        "--EXT",
                        help="extentsion for image files ",
                        default='.jpg',
                        type=str)

    args = parser.parse_args()
    tiff_dir = args.TIFFDIR
    image_dir = args.IMAGEDIR
    class_name = args.CLASSNAME
    apeer_class_name = args.APEERCLASSNAME

    check_tiff_path_format_and_tiff_image_pairs(tiff_dir=tiff_dir,
                                                image_dir=image_dir,
                                                class_name=class_name,
                                                apeer_class_name=apeer_class_name,
                                                ext=args.EXT)


if __name__ == '__main__':
    main()




