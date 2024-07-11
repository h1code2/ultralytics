#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 18:00
# @Author  : h1code2
# @File    : ll.py
# @Software: PyCharm

from tqdm import tqdm
import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

import inspect
from ultralytics.data.converter import convert_dota_to_yolo_obb


def CollateDataset(
        image_dir, label_dir, val_size=0.2, random_state=42
):  # image_dir:图片路径  label_dir：标签路径
    if not os.path.exists("./my_data"):
        os.makedirs("./my_data")

    images = []
    labels = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        ext = os.path.splitext(image_name)[-1]
        label_name = image_name.replace(ext, ".txt")
        label_path = os.path.join(label_dir, label_name)
        # 增加自定义逻辑
        if not Path(label_path).read_text().strip():
            print(f"图片({image_path})还未标注跳过")
            continue
        if not os.path.exists(label_path):
            print("there is no:", label_path)
        else:
            images.append(image_path)
            labels.append(label_path)
    train_data, test_data, train_labels, test_labels = train_test_split(
        images, labels, test_size=val_size, random_state=random_state
    )

    destination_images = "./my_data/images"
    destination_labels = "./my_data/labels"
    os.makedirs(os.path.join(destination_images, "train"), exist_ok=True)
    os.makedirs(os.path.join(destination_images, "val"), exist_ok=True)
    os.makedirs(os.path.join(destination_labels, "train_original"), exist_ok=True)
    os.makedirs(os.path.join(destination_labels, "val_original"), exist_ok=True)
    # 遍历每个有效图片路径
    for i in tqdm(range(len(train_data))):
        image_path = train_data[i]
        label_path = train_labels[i]

        image_destination_path = os.path.join(
            destination_images, "train", os.path.basename(image_path)
        )
        shutil.copy(image_path, image_destination_path)
        label_destination_path = os.path.join(
            destination_labels, "train_original", os.path.basename(label_path)
        )
        shutil.copy(label_path, label_destination_path)

    for i in tqdm(range(len(test_data))):
        image_path = test_data[i]
        label_path = test_labels[i]
        image_destination_path = os.path.join(
            destination_images, "val", os.path.basename(image_path)
        )
        shutil.copy(image_path, image_destination_path)
        label_destination_path = os.path.join(
            destination_labels, "val_original", os.path.basename(label_path)
        )
        shutil.copy(label_path, label_destination_path)


if __name__ == "__main__":
    CollateDataset("./original_data/images", "./original_data/labelTxt")

    source_file = inspect.getsourcefile(convert_dota_to_yolo_obb)
    print("\nconvert_dota_to_yolo_obb 函数所在文件的路径：", source_file)

    convert_dota_to_yolo_obb(r'/Users/h1code2/PycharmProjects/ultralytics/my_data')
