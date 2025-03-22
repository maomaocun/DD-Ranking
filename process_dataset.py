import os
from shutil import move

# 验证集路径
val_dir = "/home/tiger/Documents/NCFM/DD-Ranking/dataset/tinyimagenet/val"
annotations_file = os.path.join(val_dir, "val_annotations.txt")

# 创建类别文件夹
with open(annotations_file, 'r') as f:
    for line in f.readlines():
        img_name, class_id = line.split()[:2]
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        move(os.path.join(val_dir, "images", img_name), os.path.join(class_dir, img_name))

# 删除原始 images 文件夹
os.rmdir(os.path.join(val_dir, "images"))