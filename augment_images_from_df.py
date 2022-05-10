import pandas as pd
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import argparse

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def add_aug_prefix_to_filename(path):
    split_path = path.split('/')

    old_file_name = split_path[-1:][0]
    new_filename = 'aug-' + old_file_name

    return new_filename


def get_augmentation_from_img(augmented_img):

    bbox = np.array(augmented_img['bboxes'][0], dtype='float32')
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    output = {
        'image': augmented_img['image'],
        'xmin' : xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'class': augmented_img['class_labels'][0]
    }

    return output


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_batch_from_df(df, batch_size):
    steps = len(df)//batch_size
    remainder = len(df)%batch_size
    for i in range(0, steps):
        lesser_idx = i * batch_size
        greater_idx = (i + 1) * batch_size

        yield df.iloc[lesser_idx:greater_idx]
    yield df.iloc[len(df) - remainder:len(df)]


def save_image(image_arr, image_path):
    im = Image.fromarray(image_arr.astype(np.uint8))
    im.save(image_path)


def augment_dataframe_and_save_image(df, transformer, save_dir):
    value_list = []
    for i in range(0, len(df)):
        img_arr = load_image(df['path'].iloc[i])
        class_label = df['class'].iloc[i]
        bbox = df[['xmin', 'ymin', 'xmax', 'ymax']].iloc[i].astype('float32')
        image_path = df['path'].iloc[i]

        new_filename = add_aug_prefix_to_filename(image_path)
        new_path = os.path.join(save_dir, new_filename)
        augmentation = transformer(image=img_arr, bboxes=[bbox], class_labels=[class_label])

        augmented_output = get_augmentation_from_img(augmentation)
        width, height, _= augmented_output['image'].shape
        save_image(augmented_output['image'], new_path)
        value = (augmented_output['xmin'],
                 augmented_output['ymin'],
                 augmented_output['xmax'],
                 augmented_output['ymax'],
                 width,
                 height,
                 new_path,
                 augmented_output['class']
                 )
        value_list.append(value)
    return value_list


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


def augment_batch_from_df(df_save_path, output_dir):
    df = pd.read_pickle(df_save_path)
    df_batch_gen = get_batch_from_df(df, 16)

    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'path', 'class']
    value_list = []

    while True:
        try:
            batch_df = next(df_batch_gen)

            transform = A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=10, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ], p=0.5),
                A.HueSaturationValue(p=0.3),
            ],
                bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels']))

            value_list.extend(augment_dataframe_and_save_image(batch_df, transform, save_dir=output_dir))
        except StopIteration:
            break

    aug_df = pd.DataFrame(value_list, columns=columns)
    aug_df = format_df(aug_df)
    df_save_path = os.path.join(output_dir, 'df.dat')
    pd.to_pickle(aug_df, df_save_path)

    return aug_df


def main():
    parser = argparse.ArgumentParser(description="loads a df with image paths to be augmented",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--DFSAVEPATH',
        help='path to image data... jps and xml files.',
        type=str,
    )
    parser.add_argument(
        '-o', '--OUTPUTDIR',
        help='path to the dir where the images and new df will be saved',
        type=str,
    )
    args = parser.parse_args()
    augment_batch_from_df(df_save_path=args.DFSAVEPATH, output_dir=args.OUTPUTDIR)

#TODO find out best augmentation practices

