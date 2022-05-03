"""
usage: Reshapes images so that they are compatible with our model to train. will return the images in
(height, width, num_channels) format


optional arguments:
    -h, --help            show this help message and exit
    -i, --dir
    dir to where the image files are located. we assume that the dir is named after the class
    -he, --height
    height of the reshape
    -w, --width
    width of the rehape
    -o, --output
    output dir defualt is to overwrite the files

"""
import os
import argparse
import cv2
import glob


def reshape_images(IMAGE_DIR, OUTPUT_DIR, height, width):

    jpg_glob_path = os.path.join(IMAGE_DIR, '*[.jpg][.png][.jpeg]*')
    image_file_path_list = glob.glob(jpg_glob_path)

    # gets in input from user before overwriting or reshaping data. displays image dir and if output dir
    def confirm_reshape():
        def do_files_exist():
            exists_count = 0
            for img_path in image_file_path_list:
                filename_with_extension = img_path.split('/')[-1:][0]
                if os.path.exists(os.path.join(OUTPUT_DIR, filename_with_extension)):
                    exists_count += 1
            if exists_count > 0:
                print(f'Of {len(image_file_path_list)} image(s) There are {exists_count} existing files with the same name')
                print('Continuing will overwrite existing files')
        if OUTPUT_DIR:
            print(f'Resziing images from \n{IMAGE_DIR}\n'
                  f'and saving to \n{OUTPUT_DIR}\n with shape {height, width}')
            do_files_exist()
            while True:
                user_input = input('Continue (y/n)...')
                if user_input == 'y':
                    return True
                elif user_input == 'n':
                    print('Aborting reshape')
                    return False
                else:
                    continue
        else:
            print(f'Resziing images from {IMAGE_DIR} \n'
                  f' and OVERWRITING to {IMAGE_DIR} with shape ({height, width})')
            while True:
                user_input = input('Continue (y/n)...')
                if user_input == 'y':
                    return True
                elif user_input == 'n':
                    print('Aborting reshape')
                    return False
                else:
                    continue
    if confirm_reshape():
        for img_path in image_file_path_list:

            filename_with_extension = img_path.split('/')[-1:][0]
            img = cv2.imread(img_path)
            print(f'Reshaping img: {filename_with_extension} of shape {img.shape}.... to {height, width, img.shape[2]}\n')
            reshaped_img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
            if OUTPUT_DIR:
                image_full_path = os.path.join(OUTPUT_DIR, filename_with_extension)

            else:
                image_full_path = img_path
            cv2.imwrite(image_full_path, reshaped_img)


def main():
    parser = argparse.ArgumentParser(
        description="Reshaping images.. can save to a new directory or overwrite existing")
    parser.add_argument('-i',
                        '--IMAGEDIR',
                        help='path to the directory where our image files to be reshaped are stored',
                        type=str
    )
    parser.add_argument('-he',
                        '--HEIGHT',
                        help='height to use for our reshape',
                        type=int,
                        )
    parser.add_argument('-w',
                        '--WIDTH',
                        help='height to use for our reshape',
                        type=int,
                        )
    parser.add_argument('-o',
                        '--OUTPUTDIR',
                        help='if we want to keep the orginal file and save to a different directory',
                        type=str,
                        default=''
                        )

    args = parser.parse_args()

    reshape_images(IMAGE_DIR=args.IMAGEDIR, height=args.HEIGHT, width=args.WIDTH, OUTPUT_DIR=args.OUTPUTDIR)


if __name__ == '__main__':
    main()