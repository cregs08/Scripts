
""" usage: create_mask_image_pairs.py [-h] [-df DFSAVEPATH] [-b, BATCHSIZE] [-oi, IMAGEDIR] [-om, MASKDIR]
[-odf DFOUTPUTDIR]
loads a dataframe containing the image paths mask paths, class names, labels of a pathified image set
then saves the images and masks to the corresponding image and mask outputdir
saves a df with the image paths, mask paths, class names and labels

optional arguments:
    -h, --help            show this help message and exit
    -df DFSAVEPATH, --DFSAVEPATH
                        Path to the dataframe where our patchified images are stored.
    -b BATCHSIZE, --BATCHSIZE
                        batch size for the gen default is 16
    -oi IMAGEOUTPUTDIR, --IMAGEOUTPUTDIR
                        dir where the images are to be saved
    -om MASKOUTPUTDIR, --MASKOUTPUTDIR
                        dir where the masks are stored
    -odf DFOUTPUTDIR, --DFOUTPUTDIR
                        save dir for the df. default is root dir for images/masks


"""

import pandas as pd
import albumentations as A
import cv2
import os
import random
import matplotlib.pyplot as plt
import argparse


def get_batch_from_df(df, batch_size):
    steps = len(df)//batch_size
    remainder = len(df)%batch_size
    for i in range(0, steps):
        lesser_idx = i * batch_size
        greater_idx = (i + 1) * batch_size

        yield df.iloc[lesser_idx:greater_idx]
    yield df.iloc[len(df) - remainder:len(df)]


def load_image(image_path):
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    return mask


def save_image(image, image_path, image_output_dir):
    orig_file_name = image_path.split('/')[-1:][0]
    aug_file_name = 'aug_' + orig_file_name
    save_path = os.path.join(image_output_dir, aug_file_name)
    cv2.imwrite(save_path, image)

    return save_path


def save_mask(mask, mask_path, mask_output_dir):
    orig_file_name = mask_path.split('/')[-1:][0]
    aug_file_name = 'aug_' + orig_file_name
    save_path = os.path.join(mask_output_dir, aug_file_name)
    cv2.imwrite(save_path, mask)

    return save_path


def augment_image_and_mask(image, mask, transformer):

    augmentation = transformer(image=image, mask=mask)
    aug_image = augmentation['image']
    aug_mask = augmentation['mask']

    return aug_image, aug_mask


def load_transform_and_save_image_and_mask_batch(image_paths,
                                                 mask_paths,
                                                 labels,
                                                 transformer,
                                                 image_output_dir,
                                                 mask_output_dir):
    image_save_paths = []
    mask_save_paths = []
    #not sure if labels will have same idx so we just store them here
    new_labels = []
    for image_path, mask_path, label in zip(image_paths, mask_paths, labels):
        image_arr = load_image(image_path)
        mask_arr = load_mask(mask_path)
        aug_image, aug_mask = augment_image_and_mask(image_arr, mask_arr, transformer)

        image_save_path = save_image(aug_image, image_path, image_output_dir)
        mask_save_path = save_mask(aug_mask, mask_path, mask_output_dir)

        image_save_paths.append(image_save_path)
        mask_save_paths.append(mask_save_path)
        new_labels.append(label)

    return image_save_paths, mask_save_paths, new_labels



# finish the load batch method and test to see if image and mask files are in sync
# play with augmentations

def plot_random_pairs(aug_image_paths, aug_mask_paths):
    random.seed(42)
    randomlist = random.sample(range(0, len(aug_image_paths)), 10)

    for idx in randomlist:
        for i in range(0,1):
            image = cv2.imread(aug_image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(aug_mask_paths[idx])

            print(aug_image_paths[idx])
            print(aug_mask_paths[idx])

            plt.subplot(1,2,1)
            #plt.title(df['labels'].iloc[idx])
            plt.imshow(image)
            plt.subplot(1,2,2)
            #plt.title(df['labels'].iloc[idx])
            plt.imshow(mask[:,:,0])
            plt.show()

def get_data(batch_df, transform, image_output_dir, mask_output_dir):

    image_paths, mask_paths, labels = batch_df['image_paths'], batch_df['mask_paths'], batch_df['labels']
    aug_image_paths, aug_mask_paths, aug_labels = \
        load_transform_and_save_image_and_mask_batch(image_paths=image_paths,
                                                     mask_paths=mask_paths,
                                                     labels=labels,
                                                     transformer=transform,
                                                     image_output_dir=image_output_dir,
                                                     mask_output_dir=mask_output_dir)

    return aug_image_paths, aug_mask_paths, labels


def make_and_save_df(values, output_dir, class_name, cols=['image_paths', 'mask_paths', 'labels']):
    data = dict(zip(cols, values))
    df = pd.DataFrame(data)
    df['class_name'] = class_name
    file_name = class_name + '_image_mask_patchified_aug.dat'
    save_path = os.path.join(output_dir, file_name)
    pd.to_pickle(df, save_path)
    return df


def apply_augmentations_to_batch(df_save_path, batch_size, image_output_dir, mask_output_dir, df_output_dir):
    df = pd.read_pickle(df_save_path)
    df_batch_gen = get_batch_from_df(df, batch_size)

    first_transform = A.Compose([
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.MotionBlur(p=.3),
            A.MedianBlur(blur_limit=9, p=.3),
            A.Blur(blur_limit=15, p=.3),
        ], p=0.8),
        A.Affine(scale=1, shear=[-10, 10], translate_percent=.01, interpolation=cv2.INTER_NEAREST, fit_output=True, p=.2),
        A.GaussNoise(p=.5),
        A.ElasticTransform(p=.2)
    ],
        )
    second_transform = A.Compose([
        #A.ShiftScaleRotate(),
        A.RGBShift(p=1),
        A.Blur(p=1),
        A.GaussNoise(p=1),
        A.ElasticTransform(p=1),
    ])

    final_aug_image_save_paths = []
    final_aug_mask_save_paths = []
    final_labels = []
    class_name = None

    while True:
        try:
            batch_df = next(df_batch_gen)
            class_name = batch_df['class_name'].iloc[0]
            aug_image_paths, aug_mask_paths, labels = get_data(batch_df,
                                                               second_transform,
                                                               image_output_dir=image_output_dir,
                                                               mask_output_dir=mask_output_dir)

            final_aug_image_save_paths.extend(aug_image_paths)
            final_aug_mask_save_paths.extend(aug_mask_paths)
            final_labels.extend(labels)

        except StopIteration:
            break


    values = (final_aug_image_save_paths, final_aug_mask_save_paths, final_labels)
    df = make_and_save_df(values, df_output_dir, class_name)

    return df

    #plot_random_pairs(final_aug_image_save_paths, final_aug_mask_save_paths)


df_save_path = \
    "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/test-output/Cabernet-Sauvignon_image_mask_patchified.dat"
df = pd.read_pickle(df_save_path)

df_batch_gen = get_batch_from_df(df, 16)
df_output_dir = "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/test-output"

image_output_dir = "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/test-output/Augmented-Images-Masks/Aug-Images/Cabernet-Sauvignon"
mask_output_dir = "/Users/cole/PycharmProjects/UsefulScripts/Wine-Leaves/test-output/Augmented-Images-Masks/Aug-Masks/Cabernet-Sauvignon"

batch_size = 16


#we want our dataset to contain augmented images and orginal images
def main():
    parser = argparse.ArgumentParser(
        description="Loading a model and then using it t predict on images")

    parser.add_argument("-df",
                        "--DFSAVEPATH",
                        help="path to where the df containing the patched images is stored ",
                        type=str)
    parser.add_argument("-b",
                        "--BATCHSIZE",
                        help="batch size",
                        type=int)
    parser.add_argument("-oi",
                        "--IMAGEOUTPUTDIR",
                        help="directory to where the images will be saved",
                        type=str)
    parser.add_argument("-om",
                        "--MASKOUTPUTDIR",
                        help="directory to where the masks will be saved",
                        type=str)
    parser.add_argument("-odf",
                        "--DFOUTPUTDIR",
                        nargs='?',
                        help="directory to where the df containing the paths of our images is stored",
                        type=str)

    args = parser.parse_args()

    if not args.DFOUTPUTDIR:
        print(args.IMAGEOUTPUTDIR)
        df_output_dir = os.path.dirname(args.IMAGEOUTPUTDIR)
    else:
        df_output_dir = args.DFOUTPUTDIR

    apply_augmentations_to_batch(df_save_path=args.DFSAVEPATH,
                                      batch_size=args.BATCHSIZE,
                                      image_output_dir=args.IMAGEOUTPUTDIR,
                                      mask_output_dir=args.MASKOUTPUTDIR,
                                      df_output_dir=df_output_dir)

if __name__ == '__main__':
    main()

