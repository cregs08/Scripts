""" usage: format_image_mask_files [-h] [-t, TIFFDIR] [-om MASKOUTPUTDIR]  [-a APEERCLASSNAME] [-c CLASSNAME]


renames our tiff files into the following format:
class-name_mask_file_num.ext
Saves the renamed files to maskoutput dir
#expected tiff format:
#Cabernet-Sauvignon-0_Cabernet.ome.tiff

optional arguments:
    -h, --help            show this help message and exit
    -t TIFFDIR, --TIFFDIR
                        dir where the masks are stored as tiff files
    -om MASKOUTPUTDIR, --MASKOUTPUTDIR
                        dir where the masks are stored
    -a APEERCLASSNAME, --APEERCLASSNAME
                        name of our class from apeer. now its Cabernet. in the future we shouldn't be lazy and name them
                        properly
    -c CLASSNAME, --CLASSNAME
                        name of the class. default is the name of the tiff_file_dir

"""

import os
import cv2
import re
import glob
import argparse


#takes in a tiff file in the following format: Cabernet-Sauvignon_0_Cabernet.ome.tiff
def rename_tiff_file(tiff_file, class_name, ext='.jpg'):
    file_num = get_digits_from_file(tiff_file)
    mask_filename = class_name + '_mask_' + str(file_num) + ext

    return mask_filename


def get_digits_from_file(file):
    return int(''.join([s for s in file if s.isdigit()]))


#takes in a tiff path and sees if the filename is the correct format. appends the full path if it is not.
def get_incorrect_formatted_tiff_files(tiff_files, class_name, apeer_class_name):
    regex_pattern = "[0-9]+"
    extenstiion = '.ome.tiff'
    correct_format = class_name + '_' + regex_pattern + '_' + apeer_class_name + extenstiion
    print(correct_format)
    rex = re.compile(correct_format)

    incorrect_format = []
    for tiff in tiff_files:
        filename = os.path.basename(tiff)
        if not rex.match(filename):
            incorrect_format.append(tiff)
    return incorrect_format


def check_num_incorrect_format(incorrect_format):
    if len(incorrect_format) > 0:
        print('The following files are not formatted correctly.. ')
        for f in incorrect_format:
            print(f)
        raise ValueError
    print('All files are formatted correctly!!\n')
    return True


def confirm_mask_outputdir(mask_output_dir, sample_tiff, sample_new_path):
    if not os.path.isdir(mask_output_dir):
        print("directory not found.. making dir")
        os.mkdir(mask_output_dir)
    print('Confirming output dir for masks..')
    print(f'Renaming tiff files from:\n{sample_tiff} To:\n{sample_new_path}')
    print(f'Rename and save tiff files to the following directory?\n{mask_output_dir}'
          f'\nThis will overwrtie existing files.')
    while True:
        user_input = input('(y/n)')
        if user_input == 'y':
            return True
        if user_input == 'n':
            return False


def rename_and_save_tiff_files(mask_tiff_paths, class_name, mask_output_dir):
    new_mask_paths = []
    for tiff_path in mask_tiff_paths:
        tiff_filename = os.path.basename(tiff_path)
        new_mask_file_name = rename_tiff_file(tiff_filename, class_name)
        new_mask_path = os.path.join(mask_output_dir, new_mask_file_name)
        save_new_mask_path(tiff_path, new_mask_path)

    return new_mask_paths


def save_new_mask_path(old_path, new_mask_path):
    mask = cv2.imread(old_path)
    if not cv2.imwrite(new_mask_path, mask):
        raise Exception("Could not write image")


def do_checks_and_rename_and_save_tiff_paths(tiff_mask_dir, mask_output_dir, apeer_class_name, class_name=None):

    if not class_name :
        class_name = os.path.basename(tiff_mask_dir)
    glob_path = os.path.join(tiff_mask_dir, "*.tiff")
    tiff_paths = glob.glob(glob_path)

    sample_tiff = os.path.basename(tiff_paths[0])
    sample_new_path = rename_tiff_file(sample_tiff, class_name)
    print(sample_new_path)
    incorrectly_formatted_files = get_incorrect_formatted_tiff_files(tiff_paths, class_name, apeer_class_name)

    mask_output_dir = os.path.join(mask_output_dir, class_name)
    if check_num_incorrect_format(incorrectly_formatted_files) and confirm_mask_outputdir(mask_output_dir,
                                                                               sample_tiff=sample_tiff,
                                                                               sample_new_path=sample_new_path):
        print('\nCHECKS DONE! Renaming and saving files!\n')
        rename_and_save_tiff_files(tiff_paths, class_name, mask_output_dir=mask_output_dir )


def main():
    parser = argparse.ArgumentParser(
        description="rename and saving mask files")

    parser.add_argument("-t",
                        "--TIFFDIR",
                        help="path to where the masks are stored as .tiff files",
                        type=str)
    parser.add_argument("-om",
                         "--MASKOUTPUTDIR",
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
    args = parser.parse_args()
    do_checks_and_rename_and_save_tiff_paths(tiff_mask_dir=args.TIFFDIR,
                                             mask_output_dir=args.MASKOUTPUTDIR,
                                             apeer_class_name=args.APEERCLASSNAME,
                                             class_name=args.CLASSNAME)


if __name__ == '__main__':
    main()
