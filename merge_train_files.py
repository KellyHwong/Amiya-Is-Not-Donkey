#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-02-20 15:40
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import shutil

DONKEY_DATA_DIR = "./data/donkey/"
RABBIT_DATA_DIR = "./data/rabbit/"


def main():
    """
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'bad': # bad 1
            categories.append(1)
        else: # good 0
            categories.append(0)
    """
    root = os.getcwd()
    os.chdir(DONKEY_DATA_DIR)
    filenames = os.listdir(".")
    for filename in filenames:
        new_filename = "donkey." + filename
        os.rename(filename, new_filename)
        # print("Moving %s" % new_filename)
        shutil.move(new_filename, "..")
    os.chdir(root)
    os.chdir(RABBIT_DATA_DIR)
    filenames = os.listdir(".")
    for filename in filenames:
        new_filename = "rabbit." + filename
        os.rename(filename, new_filename)
        # print("Moving %s" % new_filename)
        shutil.move(new_filename, "..")
    os.chdir(root)
    # os.rmdir(DONKEY_DATA_DIR)
    # os.rmdir(RABBIT_DATA_DIR)


if __name__ == "__main__":
    main()
