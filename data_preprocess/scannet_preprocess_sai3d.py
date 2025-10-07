# from SAM3D prperocess Scannet

import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio.v2 as imageio
from SensorData import SensorData
import util

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='',
                    help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=640, help='export image width')
parser.add_argument('--output_image_height', type=int, default=480, help='export image height')

parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
if opt.export_label_images:
    assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)


# from https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py
def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)


def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    label_mapping = None
    if opt.export_label_images:
        label_map = util.read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')

    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    scenes.reverse()
    print('Found %d scenes' % len(scenes))
    for i in range(0,len(scenes)):
        sens_file = os.path.join(opt.scannet_path, scenes[i], scenes[i] + '.sens')
        label_path = os.path.join(opt.scannet_path, scenes[i], opt.label_type)
        gen_output_path = os.path.join(opt.output_path, scenes[i])
        if opt.export_label_images and not os.path.isdir(label_path):
            print_error('Error: using export_label_images option but label path %s does not exist' % label_path)
        if not os.path.isdir(gen_output_path):
            os.makedirs(gen_output_path)
        output_label_path = os.path.join(gen_output_path, 'label')
        if opt.export_label_images and not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)

        # read and export
        sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        sd = SensorData(sens_file)
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        sd.export_color_images(gen_output_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_depth_images(gen_output_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_poses(gen_output_path, frame_skip=opt.frame_skip)
        sd.export_intrinsics(gen_output_path)

        if opt.export_label_images:

            for f in range(0, len(sd.frames), opt.frame_skip):
                label_file = os.path.join(label_path, str(f) + '.png')
                image = np.array(imageio.imread(label_file))
                image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0,
                                    preserve_range=True)
                mapped_image = map_label_image(image, label_map)
                imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)
    print('')


if __name__ == '__main__':
    main()