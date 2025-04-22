
from ultralytics import YOLO

model = YOLO("D:/K_REPO/ComV/yolo/yolo/best.pt")
class_names = model.names
trained_objects = list(model.names.values())
print(trained_objects)
