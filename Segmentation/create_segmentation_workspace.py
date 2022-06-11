""" usage: takes in the working dir and create a workspace in the following format
Orginal-Images-Masks
    Images
        Class-Name(s)
    Masks
        Class-Name(s)

Patches
    Images
        Class-Name(s)
    Masks
        Class-Name(s)
prints a dictionary containing the dir paths

optional arguments:
    -h, --help            show this help message and exit
    -d, --PROJECTDIR
        dir to project
    -c, --CLASSNAME
        name of classes uses the join method seperate multiple by '_'


"""

import os
import argparse

dir_list = ['Orignal-Images-Masks', 'Patches']
sub_dir_list = ['Images', 'Masks']


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_workspace(project_dir, class_names):

    for d in dir_list:
        current_dir = os.path.join(project_dir, d)
        make_dir_if_not_exist(current_dir)

        if d == 'Orignal-Images-Masks':
            tiff_dir = os.path.join(current_dir, 'Tiff-Files')
            make_dir_if_not_exist(tiff_dir)

        for sd in sub_dir_list:
            current_sub_dir = os.path.join(current_dir, sd)
            make_dir_if_not_exist(current_sub_dir)
            for c in class_names:
                class_name_dir = os.path.join(current_sub_dir, c)
                make_dir_if_not_exist(class_name_dir)


def main():
    parser = argparse.ArgumentParser(description="Creating workspace for segmentation projects")

    parser.add_argument("-d",
                        "--PROJECTDIR",
                        help="directory where our images will be preprocessed and training/test data will be stored",
                        type=str)
    parser.add_argument("-c",
                        "--CLASSNAMES",
                        help='names of the classes. seperated by ""_"" ')
    args = parser.parse_args()

    class_names = args.CLASSNAMES.split('_')

    make_workspace(project_dir=args.PROJECTDIR, class_names=class_names)

