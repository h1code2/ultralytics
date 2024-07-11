#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 19:29
# @Author  : h1code2
# @File    : yoloobb.py
# @Software: PyCharm


import inspect
from ultralytics.data.converter import convert_dota_to_yolo_obb

source_file = inspect.getsourcefile(convert_dota_to_yolo_obb)
print("convert_dota_to_yolo_obb 函数所在文件的路径：", source_file)

convert_dota_to_yolo_obb(r'/Users/h1code2/PycharmProjects/ultralytics/my_data')
