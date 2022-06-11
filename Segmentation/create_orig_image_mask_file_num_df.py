""" usage:
creates a df with the formtted image paths, formatted mask paths, and their respective filenumbers

expected file format: {class-name}_{image_or_mask}_{file_num}.ext
e.g: Cabernet-Sauvignon_image_0.jpg

optional arguments:
    -h, --help            show this help message and exit
    -i IMAGEDIR, --IMAGEDIR
                        Path to the folder where the image dataset is stored.
    -m MASKDIR, --MASKDIR
                        dir where the masks are stored
    -o OUTPUTDIR, --OUTPUTDIR
                        save dir for the df, default is the base dir for IMAGEDIR
    -c CLASSNAME --CLASSNAME
                        class name used for the df save_path default is basename of IMAGEDIR
    -e EXT       --EXT
                        extension for files default is jpg

"""


import os
import glob
import pandas as pd
import argparse


def check_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"PATH {path} NOT FOUND")
    print('All paths exist!!')


def get_digits_from_file(file):
    return ''.join([s for s in file if s.isdigit()])


def get_file_nums_from_paths(image_paths, mask_paths):

    image_paths.sort()
    mask_paths.sort()

    image_file_nums = [get_digits_from_file(path) for path in image_paths]
    mask_file_nums = [get_digits_from_file(path) for path in mask_paths]

    if are_valid_file_nums(image_file_nums, mask_file_nums):
        return image_file_nums
    else:
        return False


def are_valid_file_nums(image_file_nums, mask_file_nums):
    if image_file_nums == mask_file_nums:
        return image_file_nums
    raise Exception('Invalid file numbers do a format check')


def get_class_name(class_name, image_dir):
    if not class_name:
        class_name = os.path.basename(image_dir)
    return class_name


def get_df_output_dir(df_output_dir, image_dir):
    if not df_output_dir:
        df_output_dir = os.path.dirname(os.path.dirname(image_dir))
    return df_output_dir


def do_checks_make_and_save_df(image_dir, mask_dir, df_output_dir, class_name, ext='.jpg'):
    image_paths = glob.glob(os.path.join(image_dir, "*" + ext))
    mask_paths = glob.glob(os.path.join(mask_dir, "*" + ext))

    for paths in [image_paths, mask_paths]:
        check_paths(paths)
    class_name = get_class_name(class_name, image_dir)
    df_output_dir = get_df_output_dir(df_output_dir, image_dir)
    file_nums = get_file_nums_from_paths(image_paths, mask_paths)
    df = make_df(image_paths, mask_paths, file_nums, class_name)
    save_df(df, df_output_dir, class_name)

    return df


def make_df(image_paths, mask_paths, filenums, class_name):
    cols = ['image_paths', 'mask_paths', 'file_num']
    data = dict(zip(cols, [image_paths, mask_paths, filenums]))
    df = pd.DataFrame(data)
    df['class_name'] = class_name
    return df


def save_df(df, df_output_dir, class_name):
    df_save_file = class_name + "_orig_image_mask_df.dat"
    save_path = os.path.join(df_output_dir, df_save_file)

    pd.to_pickle(df, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Loading a model and then using it t predict on images")

    parser.add_argument("-i",
                        "--IMAGEDIR",
                        help="path to where the images are stored ",
                        type=str)
    parser.add_argument("-m",
                        "--MASKDIR",
                        help="path to where the masks are stored",
                        type=str)
    parser.add_argument("-c",
                        "--CLASSNAME",
                        help="name of the class default is gotten from IMAGEDIR",
                        type=str,
                        nargs='?')
    parser.add_argument("-o",
                         "--OUTPUTDIR",
                        help="save path for the dataframe",
                        type=str,
                        nargs='?')
    parser.add_argument("-e",
                        "--EXT",
                        help="extenstion of the image files. default: .jpg",
                        default='.jpg',
                        type=str)

    args = parser.parse_args()

    image_dir = args.IMAGEDIR
    mask_dir = args.MASKDIR
    class_name = args.CLASSNAME
    output_dir = args.OUTPUTDIR
    do_checks_make_and_save_df(image_dir=image_dir,
                               mask_dir=mask_dir,
                               df_output_dir=output_dir,
                               class_name=class_name)


if __name__ == '__main__':
    main()


