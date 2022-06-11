import os

working_dir = "/Users/cole/PycharmProjects/UsefulScripts/Segmentation"

dir_list = ['Orignal-Images-Masks', 'Patches']
sub_dir_list = ['Images', 'Masks']
class_names = ['Cabernet-Sauvignon']

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

for d in dir_list:
    current_dir = os.path.join(working_dir, d)
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
