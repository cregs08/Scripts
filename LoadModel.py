"""
usage: LoadModel.py loads in a model makes classification and localization predictions.

expected file format:
│  └─ my_model/
│     ├─ checkpoint/
│     └─ pipeline.config

optional arguments:
    -h, --help            show this help message and exit
    -m, --modelDir
    dir to model where config and checkpoint is stored
    -l, --labelmap
    path to where the label_map.pbtxt is stored
    -i, --IMAGEDIR
    dir to where the test images are stored
    -n, --numImages
    int of of how many images to show defualt is 1
    -c, --checkpointNum
    number of saved checkpoint filename eg CHECKPOINTDIR/ckpt+str(checkpointNum)

"""

import tensorflow as tf
import numpy as np
import os
import argparse
import glob

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def build_model_and_restore_checkpoint(MODEL_DIR, checkpoint_num):
    def restore_checkpoint(detection_model, CHECKPOINT_PATH):
        print(f'Restoring checkpoint from {CHECKPOINT_PATH}     ....')
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(CHECKPOINT_PATH).expect_partial()
        print('Checkpoint restored!!')

    PATH_TO_CFG = os.path.join(MODEL_DIR, "pipeline.config")
    print(f'getting config from {PATH_TO_CFG}...')
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoint")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ckpt-' + str(checkpoint_num))
    restore_checkpoint(detection_model, CHECKPOINT_PATH)

    return detection_model


def predict_on_image_paths(IMAGE_PATHS, detection_model, category_index,  max_boxes_to_draw, min_score_threshold):
    print(f'Drawing a total of {max_boxes_to_draw} boxes with a min score of {min_score_threshold*100}%')
    #following methods from https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    for image_path in IMAGE_PATHS:

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)
        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        img = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_boxes_to_draw,
                min_score_thresh=min_score_threshold,
                agnostic_mode=False)
        plt.figure()
        plt.imshow(img)
        print('Done')
    plt.show()


def load_model(MODEL_DIR, LABEL_MAP_PATH, IMAGE_DIR, num_images, checkpoint_num, max_boxes_to_draw, min_score_threshold):

    print(f'Loading {num_images} images from {IMAGE_DIR}....')
    IMAGE_PATHS = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))[0:num_images]
    detection_model = build_model_and_restore_checkpoint(MODEL_DIR, checkpoint_num)

    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH,
                                                                        use_display_name=True)

    predict_on_image_paths(IMAGE_PATHS, detection_model, category_index, max_boxes_to_draw, min_score_threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Loading a model and then using it t predict on images")

    parser.add_argument("-m",
                        "--modelDir",
                        help="path to the model Dir.. eg ../myResNetModel/",
                        type=str)
    parser.add_argument("-l",
                        "--labelMap",
                        help="path to the labelmap.pbtxt",
                        type=str)
    parser.add_argument("-i",
                        "--IMAGEDIR",
                        help="path to the testDir where the test images are stored ",
                        type=str)
    parser.add_argument("-n",
                        "--numImages",
                        default=1,
                        help="number of images in test dir to main predictions on",
                        type=int)
    parser.add_argument("-c",
                        "--checkpointNum",
                        help="number of saved checkpoint filename eg .../{CHECKPOINTDIR}/ckpt+str(checkpointNum)",
                        type=int)
    parser.add_argument("-b",
                        "--maxBoxesToDraw",
                        default=1,
                        help="number of max bboxes to draw for our plot",
                        type=int)
    parser.add_argument("-th",
                        "--minScoreThreshold",
                        default=.3,
                        help="minimum confidence to draw a bbox",
                        type=float)

    args = parser.parse_args()
    load_model(MODEL_DIR=args.modelDir, LABEL_MAP_PATH=args.labelMap,
               IMAGE_DIR=args.IMAGEDIR, num_images=args.numImages,
               checkpoint_num=args.checkpointNum, max_boxes_to_draw=args.maxBoxesToDraw,
               min_score_threshold= args.minScoreThreshold)


if __name__ == '__main__':
    main()

