import os
import numpy as np
import tqdm
from utee import misc
import argparse
import cv2

imagenet_urls = [
   'http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl'
]
parser = argparse.ArgumentParser(description='Extract the ILSVRC2012 val dataset')
parser.add_argument('--in_file', default='val224_compressed.pkl', help='input file path')
parser.add_argument('--out_root', default='/tmp/public_dataset/pytorch/imagenet-data/', help='output file path')
args = parser.parse_args()

d = misc.load_pickle(args.in_file)
assert len(d['data']) == 50000, len(d['data'])
assert len(d['target']) == 50000, len(d['target'])

data224 = []
data299 = []
for img, target in tqdm.tqdm(zip(d['data'], d['target']), total=50000):
    img224 = misc.str2img(img)
    img299 = cv2.resize(img224, (299, 299))
    data224.append(img224)
    data299.append(img299)
data_dict224 = dict(
    data = np.array(data224).transpose(0, 3, 1, 2),
    target = d['target']
)
data_dict299 = dict(
    data = np.array(data299).transpose(0, 3, 1, 2),
    target = d['target']
)

if not os.path.exists(args.out_root):
    os.makedirs(args.out_root)
misc.dump_pickle(data_dict224, os.path.join(args.out_root, 'val224.pkl'))
misc.dump_pickle(data_dict299, os.path.join(args.out_root, 'val299.pkl'))






