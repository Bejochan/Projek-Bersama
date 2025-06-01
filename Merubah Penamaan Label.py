import os

label_dir = 'D:/Career/Python/yolov5/Penugasan YoloV5/Datasets/labels/train'

for filename in os.listdir(label_dir):
    if filename.endswith('.png.txt'):
        new_name = filename.replace('.png.txt', '.txt')
        os.rename(os.path.join(label_dir, filename), os.path.join(label_dir, new_name))
