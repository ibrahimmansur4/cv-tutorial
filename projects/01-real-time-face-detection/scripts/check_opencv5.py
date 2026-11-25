"""Print which face-detection APIs your installed OpenCV actually has.

Run it in any environment to confirm the version claims in this project's
article yourself instead of taking them on trust:

    python scripts/check_opencv5.py
"""

import os

import cv2

print(f"cv2.__version__           = {cv2.__version__}")
print(f"has cv2.CascadeClassifier = {hasattr(cv2, 'CascadeClassifier')}")
print(f"has cv2.FaceDetectorYN    = {hasattr(cv2, 'FaceDetectorYN')}")
print(f"has cv2.xobjdetect        = {hasattr(cv2, 'xobjdetect')}")
print(f"has dnn.readNetFromCaffe  = {hasattr(cv2.dnn, 'readNetFromCaffe')}")
print(f"has dnn.readNetFromDarknet= {hasattr(cv2.dnn, 'readNetFromDarknet')}")

directory = getattr(getattr(cv2, "data", None), "haarcascades", None)
if not directory:
    print("cv2.data.haarcascades     = <missing>")
elif not os.path.isdir(directory):
    print(f"cv2.data.haarcascades     = {directory} (directory does not exist)")
else:
    xml_files = [name for name in os.listdir(directory) if name.endswith(".xml")]
    print(f"cv2.data.haarcascades     = {directory}")
    print(f"cascade XML files there   = {len(xml_files)}")
