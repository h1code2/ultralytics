#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 20:03
# @Author  : h1code2
# @File    : predict.py
# @Software: PyCharm

#
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('best.pt')
#
# results = model(
#     ['/Users/h1code2/Downloads/TT/Screenshot_2024-07-13-10-38-33-729_com.tencent.mm.jpg'])
# #
# # # Process results list
# for result in results:
#     print(result)
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     key_points = result.keypoints  # Key_points object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95
