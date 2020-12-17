import numpy as np
from glob import glob
import os
import shutil
import cv2
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="", help="where the filtered dataset is stored")
parser.add_argument("--dump_root", type=str, default="", help="Where to dump the data")
parser.add_argument("--b_filter", type=bool, default=False, help="we do not use this in our paper")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

# labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

def main():
    datadir = args.dataset

    images = glob(os.path.join(datadir, 'image_left', '*.png'))
    image_names = [os.path.basename(name) for name in images]

    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)
    label_out = os.path.join(args.dump_root, 'map_csv')
    if not os.path.exists(label_out):
        os.makedirs(label_out)
    
    for image_name in image_names:
        print(image_name)
        imgpath = os.path.join(datadir, 'image_left', image_name)
        labelpath = os.path.join(datadir, 'gt_index', image_name)
        if not os.path.exists(labelpath):
            print('not exists', labelpath)
            continue
        shutil.copy(imgpath, args.dump_root)

        label = cv2.imread(labelpath, cv2.IMREAD_GRAYSCALE)
        label = label / 10
        outpath = os.path.join(label_out, re.sub(r'\.png$', '.csv', image_name))
        
        np.savetxt(outpath, label, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
