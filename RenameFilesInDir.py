"""
usage: RenameFilesInDir.py used for renaming collected image files in the following format {Class_Name}-{num_file}.{extenstion}


optional arguments:
    -h, --help            show this help message and exit
    -d, --dir
    dir to where the image files are located. we assume that the dir is named after the class
    -c, --className
    name of the class of the image. default is the name of the dir
    -e, --ext
    extenstion default '.jpg'

"""

import os
import argparse

#test_dir = "/Users/cole/Desktop/Wine-Leaf-Varietals/Gewürztraminer"


def rename_files(image_dir, class_name, extension):
    image_file_list = os.listdir(image_dir)

    def confirm_rename():
        sample_file = class_name + '-1' + extension
        print(f'Renaming image files from {image_dir}')
        print(f'Sample file: {image_file_list[0]}\nTo be renamed to {sample_file}')
        while True:
            y_n = input('Continue (y/n)...\t')
            if y_n == 'y':
                return True
            if y_n == 'n':
                print('Rename Aborted')
                return False
            else:
                print('Invalid input')
                continue

    if confirm_rename():
        for i, img_file in enumerate(image_file_list):
            old_file_path = os.path.join(image_dir, img_file)
            new_file_name = class_name + '-' + str(i) + extension
            new_file_path = os.path.join(image_dir, new_file_name)

            os.rename(old_file_path, new_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Renaming files from a specified directory")
    parser.add_argument('-d',
                        '--dir',
                        help='path to the directory where our image files to be renamed are stored',
                        type='str'
    )
    parser.add_argument('-c',
                        '--className',
                        help='name of the class default will be name of the directory',
                        type=str,
                        default='')
    parser.add_argument('-e',
                        '--ext',
                        help='name of the file extenstion. default is jpg',
                        type=str,
                        default='.jpg')

    args = parser.parse_args()

    if args.className:
        class_name = args.className
    else:
        class_name = args.dir.split('/')[-1:][0]

    rename_files(image_dir=args.dir, class_name=class_name, extension=args.ext)
