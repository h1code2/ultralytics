#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 20:03
# @Author  : h1code2
# @File    : predict.py
# @Software: PyCharm


from ultralytics import YOLO

# Load a model
model = YOLO(r'/Users/h1code2/PycharmProjects/ultralytics/runs/obb/train/weights/best.pt')

results = model(
    ['/Users/h1code2/PycharmProjects/ultralytics/original_data/images/4ee76de8-882d-4dd9-b523-46b0dc4faebf.jpg'])
#
# # Process results list
for result in results:
    print(result)
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keyp_oints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
