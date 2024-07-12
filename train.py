#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 19:58
# @Author  : h1code2
# @File    : train.py
# @Software: PyCharm

from ultralytics import YOLO


def main():
    model = YOLO("./yolov8-obb.yaml").load("yolov8x-obb.pt")
    # model.train(data="./dota8-obb.yaml", epochs=30, imgsz=640, cache=False, batch=2, workers=4, device="mps")
    model.train(data="./dota8-obb.yaml", epochs=30, imgsz=640, cache=False, batch=2, workers=4)


if __name__ == "__main__":
    main()
