
""" usage: format_image_mask_files [-h] [-i IMAGEMASKDIR] [-im IMAGEORMASK]
        [-c CLASSNAME] [-of ORIGFORMAT] [-pf PATCHFORMAT]


checks our image or mask dir if they are in the correct format
e.g for orginal file
class-name_image_or_mask_[0-9]+.ext
Cabernet-Sauvignon_image_0.jpg
e.g for patch file
class-name_image_or_mask_[0-9]+_patch_[0-9]+.ext
Cabernet-Sauvignon_mask_0_patch_0.jpg
arguments:
    -h, --help            show this help message and exit
    -i IMAGEMASKDIR, --IMAGEMASKDIR
                        Path to the folder where the image dataset is stored.
    -im IMAGEORMASK --IMAGEORMASK
                        weather we are looking at images or masks. default will be gotten from dir images/class-name
    -c CLASSNAME, --CLASSNAME
                        name of the class. default is the name of the image dir
    -op ORIGPATTERN, --ORIGPATTERN
                        bool value to use pattern used by regex for the orginal format: class-name_image-or-mask_[0-9]+.ext
    -pp PATCHPATTERN   --PATCHPATTERN
                        bool value to use pattern used by regex for the orginal format:
                        class-name_image-or-mask_[0-9]+_patch_[0-9].ext

"""

import re
import os
import glob
import argparse


def is_image_or_mask_dir(image_mask_dir):
    image_or_mask = os.path.dirname(image_mask_dir)
    image_or_mask = os.path.basename(image_or_mask).lower()
    if image_or_mask == 'images':
        return 'image'
    if image_or_mask == 'masks':
        return 'mask'
    else:
        return False


#where rex is an re.compile(pattern) object
def check_filename_format(rex, path):
    filename = os.path.basename(path)
    if rex.match(filename):
        return True
    else:
        return False


def get_num_correct_incorrect_format_of_paths(paths, correct_format):

    rex = re.compile(correct_format)
    formatted_paths = []
    unformatted_paths = []

    for path in paths:
        if check_filename_format(rex, path):
            formatted_paths.append(path)
        else:
            unformatted_paths.append(path)
    output = {
        'correct_format': formatted_paths,
        'incorrect_format': unformatted_paths
    }
    return output


def get_correct_format(class_name, image_or_mask, regex_pattern='[0-9]+', extension='.jpg'):
    correct_format = class_name + '_' + image_or_mask + '_' + regex_pattern + extension
    return correct_format


# checks if we want to continue with incorrect formatted paths or not
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


#image_mask_paths is a list of all files ending in extension from the fiven image or mask dir, correct_format
#will be used by re.compile to search for the given format.
def check_paths(image_mask_dir, correct_format, extension='.jpg'):
    glob_path = os.path.join(image_mask_dir, "*.*")
    paths = glob.glob(glob_path)
    checked_paths = get_num_correct_incorrect_format_of_paths(paths, correct_format)
    total_paths = len(paths)
    incorrect_formatted_paths = checked_paths['incorrect_format']
    num_incorrect_formatted_paths = len(incorrect_formatted_paths)

    if num_incorrect_formatted_paths > 0:
        display_incorrect_formatted_paths(incorrect_formatted_paths, total_paths)
    else:
        print('All files formatted correctly!!!')


# checks weather image_or_mask is passed in or if to obtain it from the image_mask_dir. Then it sees if the
# image_mask_dir is valid
def check_image_mask_arg(image_or_mask, image_mask_dir):

    if image_or_mask:
        return image_or_mask
        #sees if its a valid dir
    else:
        image_or_mask = is_image_or_mask_dir(image_mask_dir)

        if not image_or_mask:
            raise Exception(f'CANT DETERMINE IF IMAGE OR MASK DIR FILL IN -IM PARAM FOR DIR {image_mask_dir}\n'
                            f'Expected format: /../Images/class-name')
        return image_or_mask


def check_class_name(class_name, image_mask_dir):
    if not class_name:
        class_name = os.path.basename(image_mask_dir)
    return class_name


def get_format(orig_format, patch_format, class_name, image_or_mask, ext):
    regex_pattern = "[0-9]+"
    if orig_format:
        correct_format = class_name + '_' + image_or_mask + '_' + regex_pattern + ext
    elif patch_format:
        correct_format = class_name + '_' + image_or_mask + '_' + regex_pattern + '_patch_' + regex_pattern + ext
    else:
        raise Exception('No format type selected..')
    return correct_format


def check_args(image_or_mask, image_mask_dir, class_name, orig_format, patch_format, ext):
    checked_image_or_mask = check_image_mask_arg(image_or_mask, image_mask_dir)
    print(checked_image_or_mask)
    checked_class_name = check_class_name(class_name, image_mask_dir)
    correct_format = get_format(orig_format, patch_format, checked_class_name, checked_image_or_mask, ext)
    checked_output = {
        'image_or_mask': checked_image_or_mask,
        'class_name': checked_class_name,
        'correct_format': correct_format,
    }

    return checked_output


def display_selected_args(image_or_mask, image_mask_dir, correct_format):
    print(f'######\nChecking format for the {image_or_mask} files in {image_mask_dir}\n'
          f'Seleceted format were checking for {correct_format}\n#######')


def check_args_and_check_paths(image_or_mask, image_mask_dir, class_name, orig_format, patch_format, ext):
    checked_args = check_args(image_or_mask=image_or_mask,
                              image_mask_dir=image_mask_dir,
                              class_name=class_name,
                              orig_format=orig_format,
                              patch_format=patch_format,
                              ext=ext)
    correct_format = checked_args['correct_format']
    display_selected_args(image_or_mask, image_mask_dir, correct_format)
    check_paths(image_mask_dir, correct_format)

    return


def main():
    parser = argparse.ArgumentParser(
        description="formatting our image and mask files")

    parser.add_argument("-d",
                        "--IMAGEMASKDIR",
                        help="path to where the images or masks are stored ",
                        type=str)
    parser.add_argument("-of",
                         "--ORIGFORMAT",
                        help="bool value to use pattern used by regex for the orginal file format: "
                             "class-name_image-or-mask_[0-9]+.ext",
                        type=bool,
                        default=False)
    parser.add_argument("-pf",
                        "--PATCHFORMAT",
                        help= " bool value to use pattern used by regex for the orginal format:"
                            "class-name_image-or-mask_[0-9]+_patch_[0-9].ext",
                        type=bool,
                        default=False
                        )
    parser.add_argument("-im",
                        "--IMAGEORMASK",
                        help="weather we are looking at images or masks. "
                             "default will be gotten from dir images/class-name",
                        type=str,
                        nargs='?')
    parser.add_argument("-c",
                        "--CLASSNAME",
                        help="name of the class. defualt is name of the base dir for IMAGEDIR",
                        nargs='?',
                        type=str)
    parser.add_argument("-e",
                        "--EXT",
                        help='extension type of the image files. default is .jpg',
                        default='.jpg')

    args = parser.parse_args()
    image_mask_dir = args.IMAGEMASKDIR
    image_or_mask = args.IMAGEORMASK
    class_name = args.CLASSNAME
    ext = args.EXT
    orig_format = args.ORIGFORMAT
    patch_format = args.PATCHFORMAT

    #checking our args so that we have the correct format to check the paths with.
    check_args_and_check_paths(image_or_mask=image_or_mask,
                              image_mask_dir=image_mask_dir,
                              class_name=class_name,
                              orig_format=orig_format,
                              patch_format=patch_format,
                              ext=ext)


if __name__ == '__main__':
    main()
