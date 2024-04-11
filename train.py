'''
How to run this
python train.py --epochs 10 --imgsz 416 --task_name "test_run"

#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format
'''

import argparse
from ultralytics import YOLO
from allegroai import Task

# Set up argparse for command line arguments
parser = argparse.ArgumentParser(description='Train a YOLO model with custom parameters.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--imgsz', type=int, default=416, help='Image size')
parser.add_argument('--optimizer', type=str, default='auto', help='Pick your optimizer SGD/Adam/AdamW. Default is auto')
parser.add_argument('--task_name', type=str, default='yolov8n-seg', help='Task name for ClearML')
args = parser.parse_args()

# Step 1: Creating a ClearML Task
task = Task.init(
    project_name="mv-ci/as-color-extraction-segment",
    task_name=args.task_name
)

model = YOLO("yolov8n-seg.pt")  # load a pretrained segmentation model for finetuning

# Use the model
training_results = model.train(data="trainconfig.yaml", epochs=args.epochs, imgsz=args.imgsz, optimizer = args.optimizer, device=[0, 1, 2, 3, 4, 5, 6, 7])  # train the model
optimizer = model["optimizer"]
optimizer.train()
metrics = model.val()  # evaluate model performance on the validation set
optimizer.val()
print("Training Results .....")
print(training_results)
print("Metrics")
print(metrics)

# Connect arguments and metrics with ClearML task
task_arguments = dict(data="trainconfig.yaml", epochs=args.epochs, imgsz=args.imgsz, optimizer = args.optimizer, val_metrics=metrics)
task.connect(task_arguments)

