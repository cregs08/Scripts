""" usage: takes in a dir and creates a workspace for tensorflow object recognition in the following format
├─ annotations/
├─ exported-models/
├─ images/
│  ├─ test/
│  └─ train/
├─ models/
├─ pre-trained-models/
└─ README.md

prints a dictionary containing the dir paths

optional arguments:
  -h, --help            show this help message and exit
  -d, --projectDir
    dir to .../workspace/project

"""

import os
import argparse


def make_dirs(project_dir):
    dir_name_list = ['annotations', 'exported-models', 'images', 'models', 'pre-trained-models', 'addons', 'labelimg-df']

    dirs = {}
    try:
        os.mkdir(project_dir)
    except:
        FileExistsError


    for dir_name in dir_name_list:
        dir_path = os.path.join(project_dir, dir_name)
        try:
            os.mkdir(dir_path)
        except:
            FileExistsError
        dirs[dir_name.upper()] = dir_path
        if dir_name == 'images':
            for train_test in ['train', 'test']:
                image_dir_path = os.path.join(dir_path, train_test)
                try:
                    os.mkdir(image_dir_path)
                except:
                    pass
                dirs[train_test.upper()] = image_dir_path
    print("Workspace created with the following directories:\n", dirs)


def main():

    parser = argparse.ArgumentParser(
        description="Creating a workspace. Defined above")

    parser.add_argument("-d",
                        "--projectDir",
                        help="path to where we want our project stored e.g .../Tensorflow/workspace/{PROJECT_DIR}",
                        type=str)
    args = parser.parse_args()
    make_dirs(args.projectDir)


if __name__ == '__main__':
    main()

