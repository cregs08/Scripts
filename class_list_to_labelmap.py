#from https://lifesaver.codes/answer/how-to-make-label-map-pbtxt-for-object-detction-1601
#From user @sayakpaul?

#modified using arg parse


from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import argparse
import os


def convert_classes(class_list, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(class_list, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


def write_and_save_labelmap(class_list, output_dir):
    text = convert_classes(class_list)
    output_path = os.path.join(output_dir, "label_map.pbtxt")
    with open(output_path, 'w') as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(
        description="Creating a label_map.pbtxt file using a passed list with classes")

    parser.add_argument("-c",
                        "--CLASSLIST",
                        help="delimited list of classes by (',')",
                        type=str,
                        )
    parser.add_argument("-o",
                        "--OUTPUTDIR",
                        help="Directory to where our label map will be saved",
                        type=str)
    args = parser.parse_args()
    class_list = args.CLASSLIST.split(',')
    write_and_save_labelmap(class_list=class_list, output_dir=args.OUTPUTDIR)


if __name__ == '__main__':
    main()
