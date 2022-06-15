""" usage: format_image_mask_files [-h] [-df DFPATH] pd [PATCHDIR] [-om MASKOUTPUTDIR] [odf DFOUTPUTDIR] [ph PATCHHEIGHT]
 [pw PATCHWIDTH]

formats our image files into the following format:
class-name_image_file_num.ext
formats our tiff files into the following format:
class-name_mask_file_num.ext
save our image, mask paths and filenumbers as df


optional arguments:
    -h, --help            show this help message and exit
    -df -DFPATH, --DFPATH
                        df save path where our image and mask paths are stored
    -pd PATCHDIR, --PATCHDIR
                        directory containing the sub dirs Images/class-name and Masks/class-name.
    -b BATCHSIZE, --BATCHSIZE
                        batch size for the gen default is 16
    -ph PATCHHEIGHT --PATCHHEIGHT
                        height used in the patchify module. default is 512
    -pw PATCHWIDTH --PATCHWIDTH
                        width used in the patchify module. default is 512
    -s STEP --STEP
                        step used for our patches default is PATCHHEIGHT to avoid overlap

    -om MASKOUTPUTDIR, --MASKOUTPUTDIR
                        dir where the masks are stored. default is patchdir/masks/class-name

    -odf DFOUTPUTDIR, --DFOUTPUTDIR
                        save directory for our df. defualt is the patchdir

"""

from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from patchify import patchify
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class CustomDatagenImageMaskPatchify(Sequence):

    def __init__(self, df,
                 batch_size,
                 output_image_dir,
                 output_mask_dir,
                 patch_size_height,
                 patch_size_width,
                 patch_step,
                 shuffle=False,
                 ):
        self.df = df.copy()
        self.file_numbers = self.df['file']
        self.class_name = df['class_name'].iloc[0]
        self.batch_size = batch_size
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.patch_size_height = patch_size_height
        self.patch_size_width = patch_size_width
        self.patch_step = patch_step
        self.shuffle = shuffle
        self.n = len(self.df)


    def load_image_as_array(self, image_path):

        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_mask_as_array(self, mask_path):
        mask = cv2.imread(mask_path, 0)
        return mask

    def patchify_image(self, image_as_arr, patch_size_height, patch_size_width, step):
        image_patches = []
        patches_images = patchify(image_as_arr, (patch_size_height, patch_size_width, 3), step=step)
        for i in range(patches_images.shape[0]):
            for j in range(patches_images.shape[1]):
                single_patch_img = patches_images[i, j, :, :]
                single_patch_img = np.array(single_patch_img, dtype='float32')
                single_patch_img = cv2.resize(single_patch_img[0], (256, 256))
                image_patches.append(single_patch_img)
        return image_patches

    def patchify_mask(self, mask_as_arr, patch_size_height, patch_size_width, step):

        mask_patches = []
        patches_mask = patchify(mask_as_arr, (patch_size_height, patch_size_width), step=step)  # Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = np.array(single_patch_mask, dtype='float32')
                single_patch_mask = cv2.resize(single_patch_mask, (256, 256))

                mask_patches.append(single_patch_mask)
        return mask_patches

    def __save_image_patches(self, image_patches, filenumber):
        save_paths = []
        for patch_num, image_patch in enumerate(image_patches):
            file_name = self.class_name + '_image_' + str(filenumber) + '_patch_' + str(patch_num) + '.jpg'
            save_path = os.path.join(self.output_image_dir, file_name)
            cv2.imwrite(save_path, image_patch)
            save_paths.append(save_path)
        return save_paths

    def __save_mask_patches(self, mask_patches, filenumber):
        save_paths = []
        for patch_num, mask_patch in enumerate(mask_patches):
            file_name = self.class_name + '_mask_' + str(filenumber) + '_patch_' + str(patch_num) + '.jpg'
            save_path = os.path.join(self.output_mask_dir, file_name)
            cv2.imwrite(save_path, mask_patch)
            save_paths.append(save_path)
        return save_paths

    def get_image_data(self, image_path_batch, file_numbers):
        image_save_paths = []
        for idx, image_path in enumerate(image_path_batch):

            image_as_arr = self.load_image_as_array(image_path)
            image_patches = self.patchify_image(image_as_arr, self.patch_size_height, self.patch_size_width, self.patch_step)
            save_paths = self.__save_image_patches(image_patches, filenumber=file_numbers.iloc[idx])
            image_save_paths.extend(save_paths)
        return image_save_paths

    def get_mask_data(self, mask_path_batch, file_numbers):
        mask_save_paths = []
        labels = []
        for idx, mask_path in enumerate(mask_path_batch):
            mask_as_arr = self.load_mask_as_array(mask_path)
            mask_patches = self.patchify_mask(mask_as_arr, self.patch_size_height, self.patch_size_width, self.patch_step)
            save_paths = self.__save_mask_patches(mask_patches, file_numbers.iloc[idx])
            mask_save_paths.extend(save_paths)

            for single_mask_patch in mask_patches:
                pixels = np.max(single_mask_patch)
                if pixels > 0:
                    labels.extend([1])
                else:
                    labels.extend([0])

        return mask_save_paths, labels

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        image_path_batch = batches['image_paths']
        mask_path_batch = batches['mask_paths']
        file_numbers = batches['file_num']
        image_save_paths = self.get_image_data(image_path_batch, file_numbers)
        mask_save_paths, labels = self.get_mask_data(mask_path_batch, file_numbers)

        return image_save_paths, mask_save_paths, labels

    def __getitem__(self, index):
        idx_min = index * self.batch_size
        idx_max = (index + 1) * self.batch_size

        batches = self.df[idx_min:idx_max]

        image_save_paths, mask_save_paths, labels = self.__get_data(batches)
        return image_save_paths, mask_save_paths, labels, [self.class_name]

    def __len__(self):
        l = self.n // self.batch_size
        if l*self.batch_size < self.n:
            l += 1
        return l

    def on_epoch_end(self):

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            state = item
            yield state


def make_root_dir_sub_dir(root_dir, sub_dir):
    new_dir = os.path.join(root_dir, sub_dir)
    if not os.path.exists(new_dir):
        try:
            os.mkdir(new_dir)
        except:
            pass
    return new_dir


def check_dirs(patch_dir, image_output_dir, mask_output_dir):
    print('Checking output dirs... Continuing will overwrite any existing files')

    user_input = True
    while user_input :
        save_images = input(f'Saving images to {image_output_dir}\nContinue (y/n)')
        if save_images == 'y':
            user_input = False
        elif save_images == 'n':
            print('Aborting...')
            return False

    user_input = True
    while user_input :
        save_masks = input(f'Saving masks to {mask_output_dir}\nContinue (y/n)')
        if save_masks == 'y':
            user_input = False
        elif save_masks == 'n':
            print('Aborting...')
            return False

    user_input = True
    while user_input :
        check_df_save_path = input(f'Saving df to {patch_dir}\nContinue (y/n)')
        if check_df_save_path == 'y':
            user_input = False
        elif check_df_save_path == 'n':
            print('Aborting...')
            return False
    return True


def make_df_from_gen(gen):

    cols = ['image_paths', 'mask_paths', 'labels']
    iterator = True
    image_paths = []
    mask_paths = []
    labels = []
    class_name = ''
    count = 0
    while iterator:
        try:
            data = next(gen)
            image_paths.extend(data[0])
            mask_paths.extend(data[1])
            labels.extend(data[2])
            class_name = data[3]
            count+=1
        except Exception as e:
            print("An exception has occured:", e)
            iterator = False

    values = [image_paths, mask_paths, labels]
    data_dict = dict(zip(cols, values))

    df = pd.DataFrame(data_dict)
    df['class_name'] = class_name[0]
    return df


def save_df(df, output_dir):
    class_name = df['class_name'].iloc[0]
    df_filename = class_name + '_image_mask_patchified.dat'
    save_path = os.path.join(output_dir, df_filename)
    pd.to_pickle(df, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Patchifying images..")

    parser.add_argument("-df",
                        "--DFSAVEPATH",
                        help="path to where the df containing the paths of the orginal images are stored ",
                        type=str)
    parser.add_argument('-pd',
                        "--PATCHDIR",
                        help='dir for our patches Patches/Images/class-name and patches/Masks/class-name',
                        type=str)
    parser.add_argument("-oi",
                        "--IMAGEOUTPUTDIR",
                        help="directory to where the images will be saved",
                        nargs='?',
                        type=str)
    parser.add_argument("-om",
                        "--MASKOUTPUTDIR",
                        help="directory to where the masks will be saved",
                        nargs='?',
                        type=str)
    parser.add_argument("-odf",
                        "--DFOUTPUTDIR",
                        nargs='?',
                        help="directory to where the df containing the paths of our images is stored",
                        type=str)
    parser.add_argument("-b",
                        "--BATCHSIZE",
                        help="batch size",
                        default=16,
                        type=int)
    parser.add_argument("-ph",
                        "--PATCHHEIGHT",
                        help='size for patch height',
                        default=512,
                        type=int)
    parser.add_argument("-pw",
                        "--PATCHWIDTH",
                        help='size for patch width',
                        default=512,
                        type=int)
    parser.add_argument("-s",
                        "--STEP",
                        help='size for patch step',
                        nargs='?',
                        type=int)

    args = parser.parse_args()

    step = args.STEP
    df_output_dir = args.DFOUTPUTDIR
    output_image_dir = args.IMAGEOUTPUTDIR
    output_mask_dir = args.MASKOUTPUTDIR
    patch_dir = args.PATCHDIR

    image_dir = make_root_dir_sub_dir(patch_dir, 'Images')
    mask_dir = make_root_dir_sub_dir(patch_dir, 'Masks')

    print(f'Loading dataframe from {args.DFSAVEPATH}..')
    print(f'Checking Image and Mask dirs...')

    orignal_image_df = pd.read_pickle(args.DFSAVEPATH)
    class_name = orignal_image_df['class_name'].iloc[0]

    if not output_image_dir:
        output_image_dir = make_root_dir_sub_dir(image_dir, class_name)
    if not output_mask_dir:
        output_mask_dir = make_root_dir_sub_dir(mask_dir, class_name)
    if not step:
        step = args.PATCHHEIGHT
    if not df_output_dir:
        df_output_dir = patch_dir


    if check_dirs(patch_dir, output_image_dir, output_mask_dir):

        #make our output image dir and mask dir if '' passed then it just saves to the pwd
        my_gen = CustomDatagenImageMaskPatchify(orignal_image_df, batch_size=5,
                                            output_image_dir=output_image_dir,
                                            output_mask_dir=output_mask_dir,
                                            patch_size_height=args.PATCHHEIGHT,
                                            patch_size_width=args.PATCHWIDTH,
                                            patch_step=step)

        gen_iter = my_gen.__iter__()
        patch_df = make_df_from_gen(gen_iter)
        print(f'Saving dataframe to...{df_output_dir}')
        save_df(patch_df, df_output_dir)


if __name__ == '__main__':
    main()

import random


def plot_random_pairs(df):
    randomlist = random.sample(range(0, len(df)), 45)
    for idx in randomlist:
        for i in range(0,1):
            image = cv2.imread(df['image_paths'].iloc[idx])
            mask = cv2.imread(df['mask_paths'].iloc[idx])
            plt.subplot(1,2,1)
            plt.title(df['labels'].iloc[idx])
            plt.imshow(image[:,:,0], cmap='gray')
            plt.subplot(1,2,2)
            plt.title(df['labels'].iloc[idx])
            plt.imshow(mask[:,:,0])
            plt.show()


#todo clean up main method bigtime
#decide whether to use just filenames in DF or full paths