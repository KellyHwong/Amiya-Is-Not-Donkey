#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-07-20 16:25
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
from PIL import Image

# data path
TRAIN_DATA_DIR = "./data/train/"


def config_env():
    print("Load Config ...")
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    return CONFIG


def main():
    """有些
    用脚本把它们删除
    """
    CONFIG = config_env()
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])

    sub_dir_name = "rabbit"  # "donkey" or "rabbit"
    filenames = os.listdir(os.path.join(TRAIN_DATA_DIR, sub_dir_name))
    error_count = 0
    for filename in filenames:
        im_path = os.path.join(os.path.join(
            TRAIN_DATA_DIR, sub_dir_name, filename))
        try:
            im = Image.open(im_path)
        except:
            error_count += 1
            print("error: %d" % error_count)
            print("Removing: ", im_path)
            os.remove(im_path)  # delete damaged image files


if __name__ == "__main__":
    main()
