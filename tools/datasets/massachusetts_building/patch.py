import argparse
import os

import numpy as np
import pandas as pd
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='original dataset directory')
    parser.add_argument('output',
                        type=str,
                        help='patched dataset directory')
    parser.add_argument('--format',
                        type=str,
                        default='png',
                        help='file format')
    parser.add_argument('--ph',
                        type=int,
                        default=300,
                        help='patch height')
    parser.add_argument('--pw',
                        type=int,
                        default=300,
                        help='patch width')
    parser.add_argument('--sy',
                        type=int,
                        default=150,
                        help='stride for axis y')
    parser.add_argument('--sx',
                        type=int,
                        default=150,
                        help='stride for axis x')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # create output path if not exists
    os.makedirs(os.path.join(args.output), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'train_labels'), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'val_labels'), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.output, args.format, 'test_labels'), exist_ok=True)

    SPLIT_KEY = 'split'
    IMAGE_PATH_KEY = '{}_image_path'.format(args.format)
    LABEL_PATH_KEY = '{}_label_path'.format(args.format)

    input_metadata_path = os.path.join(args.input, 'metadata.csv')
    output_metadata_path = os.path.join(args.output, 'metadata.csv')
    input_df = pd.read_csv(input_metadata_path)
    output_d = {
        SPLIT_KEY: [],
        IMAGE_PATH_KEY: [],
        LABEL_PATH_KEY: [],
    }

    index = 0
    for _, row in input_df.iterrows():
        split, image_path, label_path = row[SPLIT_KEY], row[IMAGE_PATH_KEY], row[LABEL_PATH_KEY]
        image_dir, label_dir = os.path.dirname(image_path), os.path.dirname(label_path)
        input_name = os.path.basename(image_path)

        input_image_path = os.path.join(args.input, image_dir, input_name)
        input_label_path = os.path.join(args.input, label_dir, input_name)

        input_image = io.imread(input_image_path)
        input_label = io.imread(input_label_path)

        if split != 'train':
            output_name = input_name

            output_image_path = os.path.join(args.output, image_dir, output_name)
            output_label_path = os.path.join(args.output, label_dir, output_name)

            output_image = input_image
            output_label = input_label
            io.imsave(output_image_path, output_image)
            io.imsave(output_label_path, output_label)

            output_d[SPLIT_KEY].append(split)
            output_d[IMAGE_PATH_KEY].append(os.path.join(image_dir, output_name))
            output_d[LABEL_PATH_KEY].append(os.path.join(label_dir, output_name))
        else:
            h, w, _ = input_image.shape
            for y1 in np.arange(0, h - args.ph + 1, args.sy):
                for x1 in np.arange(0, w - args.pw + 1, args.sx):
                    index += 1
                    output_name = '{}.png'.format(index)

                    output_image_path = os.path.join(args.output, image_dir, output_name)
                    output_label_path = os.path.join(args.output, label_dir, output_name)

                    x2, y2 = x1 + args.pw, y1 + args.ph
                    output_image = input_image[y1:y2, x1:x2, :]
                    output_label = input_label[y1:y2, x1:x2]
                    io.imsave(output_image_path, output_image, check_contrast=False)
                    io.imsave(output_label_path, output_label, check_contrast=False)

                    output_d[SPLIT_KEY].append(split)
                    output_d[IMAGE_PATH_KEY].append(os.path.join(image_dir, output_name))
                    output_d[LABEL_PATH_KEY].append(os.path.join(label_dir, output_name))

        print(f'{input_image_path} is patched')

    output_df = pd.DataFrame(output_d)
    output_df.to_csv(output_metadata_path)


if __name__ == '__main__':
    main()
