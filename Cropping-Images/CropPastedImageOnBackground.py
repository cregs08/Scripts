from PIL import Image
import random
import pandas as pd
import os
import glob
import argparse
import numpy as np
import tensorflow as tf


def create_one_hot_encoings(label_encoding_col, classes_col):
    num_classes = len(classes_col.unique())

    gt_classes_one_hot_tensors = []
    for label_id in label_encoding_col:
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.array([label_id]), dtype=np.int32)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))
    return gt_classes_one_hot_tensors


def norm_bbox_cols(norm_df):

    norm_df['xmin_norm'] = norm_df['xmin'] / norm_df['width']
    norm_df['xmax_norm'] = norm_df['xmax'] / norm_df['width']

    norm_df['ymin_norm'] = norm_df['ymin'] / norm_df['height']
    norm_df['ymax_norm'] = norm_df['ymax'] / norm_df['height']

    return norm_df


def make_bbox_cols(norm_df, bbox_formats):
    def make_bbox_array(bbox_cols):
        rows = [np.array([row]) for row in norm_df[bbox_cols].to_numpy()]
        return rows

    for bbox_cols in bbox_formats:
        norm_df[bbox_cols] = make_bbox_array(bbox_formats[bbox_cols])


def get_bbox_center_width_and_height(norm_df):

    norm_df['bbox_width'] = norm_df['xmax_norm'] - norm_df['xmin_norm']
    norm_df['bbox_height'] = norm_df['ymax_norm'] - norm_df['ymin_norm']

    norm_df['x_center'] = (norm_df['xmin_norm'] + norm_df['xmax_norm']) / 2
    norm_df['y_center'] = (norm_df['ymin_norm'] + norm_df['ymax_norm']) / 2

    return norm_df


def create_class_ids(norm_df):
    norm_df['class_label_cat'] = norm_df['class'].astype('category')
    norm_df['label_encoding'] = norm_df['class_label_cat'].cat.codes


def format_df(df):
    bbox_formats = {
        'pascal_voc_bb': ['xmin_norm', 'ymin_norm', 'xmax_norm', 'ymax_norm'],
        'coco_bb': ['xmin_norm', 'ymin_norm', 'bbox_width', 'bbox_height'],
        'tf_bb': ['ymin_norm', 'xmin_norm', 'ymax_norm', 'xmax_norm'],
        'yolo_bb' : ['x_center', 'y_center', 'bbox_width', 'bbox_height']
    }
    cols_needed_for_norm = ['xmin', 'ymin', 'xmax', 'ymax', 'height', 'width']
    norm_df = df.copy()

    for col in cols_needed_for_norm:
        norm_df[col] = norm_df[col].astype('float32')

    norm_bbox_cols(norm_df)
    get_bbox_center_width_and_height(norm_df)
    make_bbox_cols(norm_df, bbox_formats)
    create_class_ids(norm_df)
    norm_df['one_hot'] = create_one_hot_encoings(norm_df['label_encoding'], norm_df['class'])

    return norm_df

#here we get the required params to resize our cropped image. now for a min factor we are using 1/10 and max is 1/4
def get_background_and_paste_image_params(background_image_shape,
                                          min_limit_factor=.05, max_limit_factor=.4):

    background_width = background_image_shape[0]
    background_height = background_image_shape[1]

    width_min_limit = int(background_width * min_limit_factor)
    height_min_limit = int(background_height * min_limit_factor)

    width_max_limit = int(background_width * max_limit_factor)
    height_max_limit = int(background_height * max_limit_factor)

    resize_width = random.randint(width_min_limit, width_max_limit)
    resize_height = random.randint(height_min_limit, height_max_limit)

    # coords where the image will be pasted (x_upper_left, y_upper_left), or (xmin, ymin)
    rand_xmin = random.randint(0, (background_width - resize_width))
    rand_ymin = random.randint(0, (background_height - resize_height))

    #getting our xmax/ymax
    rand_xmax = rand_xmin + resize_width
    rand_ymax = rand_ymin + resize_height

    image_params = {
        'resize_width': resize_width,
        'resize_height': resize_height,
        'background_image_size': background_image_shape,
        'rand_xmin': rand_xmin,
        'rand_ymin': rand_ymin,
        'rand_xmax': rand_xmax,
        'rand_ymax': rand_ymax,
        'pasted_img_coords': [rand_xmin, rand_ymin, rand_xmax, rand_ymax]
    }

    return image_params


#checks to see if the whole picture will appear in the image, e.g will not be cropped out
#if true will return the coords to paste the image: else returns false
def check_crop_image_location(image_params):
    xmin, ymin, xmax, ymax = image_params['pasted_img_coords']

    background_image_width_max_limit, background_image_height_max_limit = image_params['background_image_size']
    resize_width, resize_height = image_params['resize_width'], image_params['resize_height']

    if xmax > background_image_width_max_limit:
        print('Image out of bounds on x-axis')
        print(f'sum of rand_x coord: {xmin} and resized width {resize_width} = {xmax}')
        return False

    if ymax > background_image_height_max_limit:
        print('Image out of bounds on y-axis')
        print(f'sum of {ymin} and {resize_height} : {ymax}')
        return False

    return True


# assuming that our cropped image path is in the following format (../.../croppedImages/classLabel-0.ext)
def get_class_label_from_cropped_image_path(cropped_image_path):
    class_label_with_extension = cropped_image_path.split('/')[-1:][0]
    class_label = class_label_with_extension.split('-')[0]
    return class_label


def get_filename_from_path(path):
    filename_with_extension = path.split('/')[-1:][0]
    filename = filename_with_extension.split('.')[0]
    return filename


def paste_cropped_image_on_background_image(cropped_image_path, background_image_path, output_dir, value_list):

    background_image = Image.open(background_image_path)
    cropped_image = Image.open(cropped_image_path)

    size_background_image = background_image.size
    image_params = get_background_and_paste_image_params(size_background_image)

    if check_crop_image_location(image_params):
        width, height = size_background_image

        #coords for our crop (bbox)
        xmin, ymin, xmax, ymax = image_params['pasted_img_coords']

        cropped_image = cropped_image.resize((image_params['resize_width'], image_params['resize_height']))

        class_label = get_class_label_from_cropped_image_path(cropped_image_path)
        filename = get_filename_from_path(cropped_image_path)

        augmented_image_save_path = os.path.join(output_dir, f'augmented-{filename}.jpg')

        value = (filename,
                 width,
                 height,
                 augmented_image_save_path,
                 class_label,
                 xmin,
                 ymin,
                 xmax,
                 ymax,
                 )
        value_list.append(value)

        background_image_with_paste = background_image.copy().convert('RGB')
        background_image_with_paste.paste(cropped_image, (xmin, ymin))
        background_image_with_paste.save(augmented_image_save_path)


#incase we have more than one file we just go with the less of the two to avoid duplicates
def get_min_length(cropped_image_file_paths, background_image_file_paths):
    length_cropped_images = len(cropped_image_file_paths)
    length_background_images = len(background_image_file_paths)

    min_length = min(length_cropped_images, length_background_images)

    min_length_cropped_image_file_paths = cropped_image_file_paths[:min_length]
    min_length_background_image_file_paths = background_image_file_paths[:min_length]

    return min_length_cropped_image_file_paths, min_length_background_image_file_paths


def make_and_save_value_list_df(value_list, output_dir):
    column_name = ['filename', 'width', 'height', 'path',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']

    df = pd.DataFrame(value_list, columns=column_name)
    df = format_df(df)
    df_save_path = os.path.join(output_dir, 'df.dat')
    pd.to_pickle(df, df_save_path)


def load_crop_and_background_images(crop_image_dir, background_image_dir, output_dir):

    crop_image_glob_path = os.path.join(crop_image_dir, '*[.jpg][.png][.jpeg]*')
    cropped_image_file_paths = glob.glob(crop_image_glob_path)

    background_image_glob_path = os.path.join(background_image_dir, '*[.jpg][.png][.jpeg]*')
    background_image_file_paths = glob.glob(background_image_glob_path)

    min_length_cropped_image_file_paths, min_length_background_image_file_paths = \
        get_min_length(cropped_image_file_paths, background_image_file_paths)
    value_list = []
    for cropped_image_path, background_image_path\
            in zip(min_length_cropped_image_file_paths, min_length_background_image_file_paths):

        paste_cropped_image_on_background_image(cropped_image_path, background_image_path, output_dir, value_list)
    make_and_save_value_list_df(value_list, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Cropping background images into 4 parts.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-c', '--CROPPEDIMAGEDIR',
        help='where our image files to be cropped are stored',
        type=str,
    )
    parser.add_argument(
        '-b', '--BACKGROUNDIMAGEDIR',
        help='where are background image files are stored',
        type=str
    )
    parser.add_argument(
        '-o', '--OUTPUTDIR',
        help='path to where our newly augmented images will be saved',
        type=str,
    )

    args = parser.parse_args()

    load_crop_and_background_images(crop_image_dir=args.CROPPEDIMAGEDIR, background_image_dir=args.BACKGROUNDIMAGEDIR,
                                    output_dir=args.OUTPUTDIR)


if __name__ == '__main__':
    main()
