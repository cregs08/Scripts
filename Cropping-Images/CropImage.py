from PIL import Image
import pandas as pd
import os
import argparse


def create_save_path_and_save_cropped_image(cropped_image, image_path, output_dir):
    filename_with_extension = image_path.split('/')[-1:][0]
    save_path = os.path.join(output_dir, filename_with_extension)
    cropped_image.save(save_path)
    print(f'####\ncropping and saving image\n {filename_with_extension} to\n{save_path}...')

    return save_path


def crop_images_from_path_and_save(image_paths, crop_coords, output_dir):
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        cropped_image = image.crop(crop_coords.iloc[i])
        create_save_path_and_save_cropped_image(cropped_image, image_path, output_dir)


def check_if_dir_exists(output_dir):
        if os.path.exists(output_dir):
            return True
        else:
            print('Output Directory not found')
            while True:
                user_input = input(f'Create directory\t{output_dir} and continue cropping? (y/n)')

                if user_input == 'y':
                    os.mkdir(output_dir)
                    return True
                elif user_input == 'n':
                    print('Aborting crop')
                    return False
                else:
                    print('Invalid character... (y/n)')
                    continue


def overwrite_files(image_paths, output_dir):
    exists_count = 0
    for image_path in image_paths:
        filename_with_extension = image_path.split('/')[-1:][0]
        if os.path.exists(os.path.join(output_dir, filename_with_extension)):
            exists_count += 1
    if exists_count > 0:
        print(f'Of {len(image_paths)} image(s) There are {exists_count}'
              f' existing files with the same name in {output_dir}\n'
              f'Continuing will overwrite existing files')
        while True:
            user_input = input('Overwrite files and continue cropping? (y/n)')
            if user_input == 'y':
                return True
            elif user_input == 'n':
                print('Aborting crop')
                return False
            else:
                print('Invalid character... (y/n)')
                continue
    return True


def confirm_crop(image_paths, output_dir):

    print(f'Checking if {output_dir} exists...')
    dir_exists = check_if_dir_exists(output_dir)
    if dir_exists:
        print(f'Output directory {output_dir} found!')
        print('Checking to see if directory is empty and to overwrite exisitng files...')
        overwrite = overwrite_files(image_paths, output_dir)

        if overwrite:
            print('Saving cropped images...')
        else:
            return False
    else:
        return False

    while True:
        user_input = input(f'Last check before cropping... Crop images from xml_df to {output_dir}? (y/n)')

        if user_input == 'y':
            return True
        elif user_input == 'n':
            print('Aborting crop')
            return False
        else:
            print('Invalid character... (y/n)')
        continue


def load_images_from_df_to_crop_and_save(df_save_path, output_dir):
    df = pd.read_pickle(df_save_path)
    bbox_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    image_paths = df['path']
    bboxes = df[bbox_cols]

    if confirm_crop(image_paths, output_dir):
        crop_images_from_path_and_save(image_paths, bboxes, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Reshaping images.. can save to a new directory or overwrite existing")
    parser.add_argument('-s',
                        '--DFSAVEPATH',
                        help='save path of or xml_df where our image paths and bbox data is stored',
                        type=str
    )
    parser.add_argument('-o',
                        '--OUTPUTDIR',
                        help='if we want to keep the orginal file and save to a different directory',
                        type=str,
                        )

    args = parser.parse_args()

    load_images_from_df_to_crop_and_save(df_save_path=args.DFSAVEPATH, output_dir=args.OUTPUTDIR)


if __name__ == "__main__":
    main()
