#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-07-20 16:25
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
from PIL import Image

# data path
TRAIN_DATA_DIR = "./data/train/"


def main():
    filenames = os.listdir(TRAIN_DATA_DIR)
    error_count = 0
    for filename in filenames:
        im_path = os.path.join(TRAIN_DATA_DIR, filename)
        try:
            im = Image.open(im_path)
        except:
            error_count += 1
            print("error: %d" % error_count)
            print(im_path)
            os.remove(im_path)


if __name__ == "__main__":
    main()
