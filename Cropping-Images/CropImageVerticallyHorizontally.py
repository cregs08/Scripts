
"""
usage: CropImageVerticallyHorizontally.py crops a background image into four parts. top-half, bottom-half, left-half,
    right-half

expected file format:
    OUTPUTDIR: ~/CroppedBackground/ClassLabel/

optional arguments:
    -h, --help            show this help message and exit
    -i, --IMAGEDIR  where the images to be cropped are
    -o, --OUTPUTDIR where the images will be saved
"""
from PIL import Image
import os
import glob
import argparse


def get_filename_from_path(path):
    filename_with_extension = path.split('/')[-1:][0]
    filename = filename_with_extension.split('.')[0]
    return filename


def make_save_path(image_dir, image_path, crop_type):
    old_file_name = get_filename_from_path(image_path)
    new_filename = crop_type + '-' + old_file_name + '.jpg'
    return os.path.join(image_dir, new_filename)


def confirm(IMAGE_DIR, OUTPUT_DIR, num_images):
    print(f'Cropping {num_images} images 4 ways for a total of {num_images * 4}\n Loading Images from {IMAGE_DIR}...'
          f'Will save into {OUTPUT_DIR}')
    while True:
        user_input = input('Continue (y/n)...')
        if user_input == 'y':
            return True
        elif user_input == 'n':
            print('Aborting crop')
            return False
        else:
            continue


def crop_horizontally_and_vertically(IMAGE_DIR, OUTPUT_DIR):
    img_glob_path = os.path.join(IMAGE_DIR, '*[.jpg][.png][.jpeg]*')
    image_file_paths = glob.glob(img_glob_path)
    num_images = len(image_file_paths)

    if confirm(IMAGE_DIR, OUTPUT_DIR, num_images):

        for image_path in image_file_paths:
            background_image = Image.open(image_path)
            background_image = background_image.copy().convert('RGB')
            image_size = background_image.size

            bottom_half_crop = background_image.crop((0, image_size[1]//2, image_size[0], image_size[1]))
            bottom_half_crop_save_path = make_save_path(OUTPUT_DIR, image_path, 'bottom')
            bottom_half_crop.save(bottom_half_crop_save_path)

            top_half_crop = background_image.crop((0,0,image_size[0], image_size[1]//2))
            top_half_crop_save_path = make_save_path(OUTPUT_DIR, image_path, 'top')
            top_half_crop.save(top_half_crop_save_path)

            left_half_crop = background_image.crop((0, 0, image_size[0]//2, image_size[1]))
            left_half_crop_save_path = make_save_path(OUTPUT_DIR, image_path, 'left')
            left_half_crop.save(left_half_crop_save_path)

            right_half_crop = background_image.crop((image_size[0]//2, 0, image_size[0], image_size[1]))
            right_half_crop_save_path = make_save_path(OUTPUT_DIR, image_path, 'right')
            right_half_crop.save(right_half_crop_save_path)
        print(f'All done! Cropped {num_images * 4} images into {OUTPUT_DIR}')


def main():
    parser = argparse.ArgumentParser(description="Cropping background images into 4 parts.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--IMAGEDIR',
        help='where our image files to be cropped are stored',
        type=str,
    )
    parser.add_argument(
        '-o', '--OUTPUTDIR',
        help='path to where our newly cropped images will be saved',
        type=str,
    )

    args = parser.parse_args()

    crop_horizontally_and_vertically(args.IMAGEDIR, args.OUTPUTDIR)


if __name__ == '__main__':
    main()

