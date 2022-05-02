from PIL import Image
import random
import pandas as pd
import os
import glob
import argparse


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
