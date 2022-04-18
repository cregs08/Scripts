""" usage: LabelimgToDF.py [-h] [-i IMAGEDIR] [-s saveDir] [-c colsToDrop] [-f -bboxFormat]
python LabelimgToDF.py -i {IMAGEDIR} -s {SAVEDIR}
saves our data in the following columns:['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax',
       'xmin_norm', 'xmax_norm', 'ymin_norm', 'ymax_norm', 'bbox_width',
       'bbox_height', 'x_center', 'y_center', 'pascal_voc_bb', 'coco_bb',
       'tf_bb', 'yolo_bb', 'class_label_cat', 'label_encoding']
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored.
  -s saveDir, --saveDir
                        dir where the df will be saved
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os
import argparse
import glob
import numpy as np
import tensorflow as tf
hh

def get_jpg_and_xml_path_lists(image_dir):
    jpg_glob_path = os.path.join(image_dir, '*.jpg')
    xml_glob_path = os.path.join(image_dir, '*.xml')

    jpg_file_path_list = glob.glob(jpg_glob_path)
    xml_file_path_list = glob.glob(xml_glob_path)

    return jpg_file_path_list, xml_file_path_list

def get_filename_from_path(path):
    # filename with file extenstion eg. xxx.jpg
    filename_with_extension = path.split('/')[-1:]
    # just file eg xxx
    filename = filename_with_extension[0].split('.')[0]
    return filename


def check_file_list_length(image_path_list, xml_path_list):
    num_images = len(image_path_list)
    num_xmls = len(xml_path_list)
    if num_images == num_xmls:
        print(f'Number of jpgs, {num_images} and number of xmls, {num_xmls} are the same')
    else:
        print(f'Number of jpgs, {num_images} and number of xmls, {num_xmls} are not  the same')


def check_matching_jpg_and_xml_file_extentions(IMAGE_DIR):

    def are_missing_pairs():
        jpg_file_path_list, xml_file_path_list = get_jpg_and_xml_path_lists(IMAGE_DIR)

        check_file_list_length(jpg_file_path_list, xml_file_path_list)

        jpg_filenames = []
        xml_filenames = []

        for jpg_path, xml_path in zip(jpg_file_path_list, xml_file_path_list):
            jpg_filenames.append(get_filename_from_path(jpg_path))
            xml_filenames.append(get_filename_from_path(xml_path))

        different_filenames = []
        for xml_file in xml_filenames:
            matching_path = os.path.join(IMAGE_DIR, xml_file + '.jpg')
            if not os.path.exists(matching_path):
                xml_path = os.path.join(IMAGE_DIR, xml_file + '.xml')
                different_filenames.append(xml_path)
                print(f'file {xml_path} does not have a matching jpg')

        for jpg_file in jpg_filenames:
            matching_path = os.path.join(IMAGE_DIR, jpg_file + '.xml')
            if not os.path.exists(matching_path):
                jpg_path = os.path.join(IMAGE_DIR, jpg_file + '.jpg')
                different_filenames.append(jpg_path)
                print(f'file {jpg_path} does not have a matching xml')

        return different_filenames

    print('Checking to see if jpgs and xml files match...')
    print(f'Loading files from {IMAGE_DIR} ......')

    # if we are missing a pair then we find which file is missing
    missing_pairs = are_missing_pairs()
    if missing_pairs:
        print(missing_pairs)
        raise FileNotFoundError
    else:
        print('All jpgs have corresponding .xml')
        print('All xml have corresponding .jpg')

        return True

### from generate tf_records.py tfod tutorial
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html


def create_xml_df(xml_file_path_list):
    xml_list = []
    for xml_file in xml_file_path_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (filename,
                     width,
                     height,
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text),
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def create_one_hot_encoings(label_encoding_col, classes_col):
    num_classes = len(classes_col.unique())

    gt_classes_one_hot_tensors = []
    for label_id in label_encoding_col:
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.array([label_id]), dtype=np.int32)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))
    return gt_classes_one_hot_tensors


def format_df(df):
    bbox_formats = {
        'pascal_voc_bb': ['xmin_norm', 'ymin_norm', 'xmax_norm', 'ymax_norm'],
        'coco_bb': ['xmin_norm', 'ymin_norm', 'bbox_width', 'bbox_height'],
        'tf_bb': ['ymin_norm', 'xmin_norm', 'ymax_norm', 'xmax_norm'],
        'yolo_bb' : ['x_center', 'y_center', 'bbox_width', 'bbox_height']
    }
    cols_needed_for_norm = ['xmin', 'ymin', 'xmax', 'ymax', 'height', 'width']
    norm_df = df.copy()

    def norm_bbox_cols(cols_needed_for_norm):
        for col in cols_needed_for_norm:
            df[col] = df[col].astype('float32')

        norm_df['xmin_norm'] = norm_df['xmin'] / norm_df['width']
        norm_df['xmax_norm'] = norm_df['xmax'] / norm_df['width']

        norm_df['ymin_norm'] = norm_df['ymin'] / norm_df['height']
        norm_df['ymax_norm'] = norm_df['ymax'] / norm_df['height']

    def get_bbox_center_width_and_height():
        norm_df['bbox_width'] = norm_df['xmax_norm'] - norm_df['xmin_norm']
        norm_df['bbox_height'] = norm_df['ymax_norm'] - norm_df['ymin_norm']

        norm_df['x_center'] = (norm_df['xmin_norm'] + norm_df['xmax_norm']) / 2
        norm_df['y_center'] = (norm_df['ymin_norm'] + norm_df['ymax_norm']) / 2

    def make_bbox_cols(bbox_formats):
        def make_bbox_array(bbox_cols):
            rows = [np.array([row]) for row in norm_df[bbox_cols].to_numpy()]
            return rows

        for bbox_cols in bbox_formats:
            norm_df[bbox_cols] = make_bbox_array(bbox_formats[bbox_cols])

    def create_class_ids():
        norm_df['class_label_cat'] = norm_df['class'].astype('category')
        norm_df['label_encoding'] = norm_df['class_label_cat'].cat.codes


    norm_bbox_cols(cols_needed_for_norm)
    get_bbox_center_width_and_height()
    make_bbox_cols(bbox_formats)
    create_class_ids()
    norm_df['one_hot'] = create_one_hot_encoings(norm_df['label_encoding'], norm_df['class'])

    return norm_df


def load_and_inspect_formatted_xml_df(save_path):
    print(f"loading dataframe from {save_path}")
    formatted_xml_df = pd.read_pickle(save_path)
    class_labels = formatted_xml_df['class'].unique()
    label_encodings = formatted_xml_df['label_encoding'].unique()
    print(formatted_xml_df.head())
    print('#####\nCOLUMNS')
    print(formatted_xml_df.columns)
    print('#####\nCLASS LABELS')
    print(class_labels)
    print('#####\nLABEL ENCODINGS')
    print(dict(zip(class_labels,label_encodings)))
    print('#####\nONE HOT ENCODINGS')
    unique_one_hot_encodings = {}
    for label in class_labels:
        label_mask = formatted_xml_df['class'] == label
        label_one_hot = formatted_xml_df[label_mask]['one_hot'].iloc[0]
        unique_one_hot_encodings[label] = label_one_hot.numpy()
    print(unique_one_hot_encodings, '\n######')


def build_format_save_xml_df_from_label_image_data(IMAGE_DIR, SAVE_DIR):

    _, xml_file_path_list = get_jpg_and_xml_path_lists(IMAGE_DIR)

    xml_df = create_xml_df(xml_file_path_list)
    formatted_df = format_df(xml_df)
    save_path = os.path.join(SAVE_DIR, 'xml_df.dat')
    print(f'Saving dataframe to {SAVE_DIR}')
    pd.to_pickle(formatted_df, save_path)

    ### inspecting dataframe
    load_and_inspect_formatted_xml_df(save_path)


def main():

    parser = argparse.ArgumentParser(description="load labelimg data into a df",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='path to labelimg data... jps and xml files.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-s', '--saveDir',
        help='path to the dir where the df and the bbox df will be saved',
        type=str,
    )

    args = parser.parse_args()

    if check_matching_jpg_and_xml_file_extentions(args.imageDir):
        build_format_save_xml_df_from_label_image_data(IMAGE_DIR=args.imageDir, SAVE_DIR=args.saveDir)

if __name__ == '__main__':
    main()




