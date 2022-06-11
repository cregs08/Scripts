
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


"""
checking format for tiff files
"""


def check_tiff_file_format(tiff_file, class_name, apeer_class_name):
    rex_pattern = "[0-9]+"
    extension = '.ome.tiff'
    correct_format = class_name + '-' + rex_pattern + '_' + apeer_class_name + extension
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
            image_not_found = f'Could not find matching pair for {tiff_file} in' \
                f'\n{image_dir}\n' \
                f'Created image file: {image_filename}'
            invalid_image_paths.append(image_not_found)
    checked_paths = {
        'valid_paths': valid_image_paths,
        'invalid_paths': invalid_image_paths
    }
    return checked_paths


#expected tiff format: {class-name}-{file_num}_{apeer_class_name}.ome.tiff
def make_tiff_filename_from_image_file(image_file, class_name, apeer_class_name):
    file_num = get_digits_from_file(image_file)
    tiff_file = class_name + '-' + file_num + '_' + apeer_class_name + '.ome.tiff'
    return tiff_file


def get_valid_invalid_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name):
    valid_tiff_paths = []
    invalid_tiff_paths = []
    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        tiff_filename = make_tiff_filename_from_image_file(image_file, class_name, apeer_class_name)
        tiff_path = os.path.join(tiff_dir, tiff_filename)
        if os.path.exists(tiff_path):
            valid_tiff_paths.append(tiff_path)
        else:
            tiff_not_found = f'\nCould not find matching pair for {image_file} in' \
                f'\n{tiff_dir}\n' \
                f'Created tiff file: {tiff_filename}\n'
            invalid_tiff_paths.append(tiff_not_found)
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

def display_invalid_paths_pairs(checked_paths):
    invalid_paths = checked_paths['invalid_paths']

    num_invalid_paths = len(invalid_paths)
    if num_invalid_paths >0:
        for path in invalid_paths:
            print(path)


def check_tiff_paths(tiff_mask_paths, class_name, apeer_class_name):
    checked_tiff_paths = get_valid_invalid_tiff_paths(tiff_mask_paths, class_name, apeer_class_name)

    invalid_tiff_paths = checked_tiff_paths['invalid_paths']
    num_invalid_tiff_paths = len(invalid_tiff_paths)
    if num_invalid_tiff_paths > 0:
        display_invalid_paths(invalid_tiff_paths)
    else:
        print('####\nAll tiff files formatted correctly!')



def check_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name):
    checked_image_to_tiff_pairs = \
        get_valid_invalid_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name)
    display_invalid_paths_pairs(checked_image_to_tiff_pairs)
    invalid = checked_image_to_tiff_pairs['invalid_paths']
    print(f'####\nFound {len(invalid)} invalid pairs for image to tiff')


def check_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext):
    checked_tiff_to_image_pairs = get_valid_invalid_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext)
    display_invalid_paths_pairs(checked_tiff_to_image_pairs)
    invalid = checked_tiff_to_image_pairs['invalid_paths']
    print(f'####\nFound {len(invalid)} invalid pairs for tiff to image')


def check_tiff_path_format_and_tiff_image_pairs(tiff_dir, image_dir, class_name, apeer_class_name, ext='.jpg'):
    tiff_glob_path = os.path.join(tiff_dir, '*.tiff')
    tiff_mask_paths = glob.glob(tiff_glob_path)
    total_mask_paths = len(tiff_mask_paths)

    image_glob_path =  os.path.join(image_dir,'*'+ext )
    image_paths = glob.glob(image_glob_path)
    total_image_paths = len(image_paths)

    display_total_paths(total_image_paths, total_mask_paths)

    check_tiff_paths(tiff_mask_paths, class_name, apeer_class_name)
    check_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name)
    check_tiff_to_image_pairs(tiff_mask_paths, image_dir, class_name, ext)

    print('\nDone checking pairs!')

tiff_dir = "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/segementation-a/Orginal-Images-Masks/Masks/Tiff-files/Cabernet-Sauvignon"
image_dir = "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/segementation-a/Orginal-Images-Masks/Images/Cabernet-Sauvignon"
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

class_name = 'Cabernet-Sauvignon'
apeer_class_name = 'Cabernet'
ext = '.jpg'
check_tiff_path_format_and_tiff_image_pairs(tiff_dir=tiff_dir,
                                            image_dir=image_dir,
                                            class_name=class_name,
                                            apeer_class_name=apeer_class_name)
# checked = get_valid_invalid_image_to_tiff_pairs(image_paths, tiff_dir, class_name, apeer_class_name)
# for path in checked['invalid_paths']:
#     print(path)
# valid_paths = checked_paths['valid_paths']
# invalid_paths = checked_paths['invalid_paths']

