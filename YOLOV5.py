import json
import os
from typing import AnyStr
import torch
import scipy.io
from yolov5 import train
from sklearn.model_selection import train_test_split
import csv


def convert_image_to_label(images_path: AnyStr, labels_path: AnyStr, mat_path: AnyStr) -> None:
    fileNames = os.listdir(images_path)
    print('There are {} images in the dataset'.format(len(fileNames)))

    sets = {}
    sets['train'], sets['test'] = train_test_split(fileNames, test_size=0.25, random_state=22)
    sets['train'], sets['val'] = train_test_split(sets['train'], test_size=1 / 3, random_state=22)

    for data in sets:
        output_file_path = os.path.join(os.path.dirname(images_path), data + '.txt')
        with open(output_file_path, 'w') as f:
            for img in sets[data]:
                img_path = os.path.join(os.getcwd(), images_path, img)
                f.write(img_path + '\n')

    print("Length of Train =", len(sets['train']))
    print("Length of Valid =", len(sets['val']))
    print("Length of test =", len(sets['test']))

    mat_data = scipy.io.loadmat(mat_path)
    labels = mat_data['labels']
    os.makedirs(labels_path, exist_ok=True)

    for i in range(labels.shape[1]):
        label = labels[0, i]
        image_name = f"image_{i:05}"
        output_file_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + '.txt')

        with open(output_file_path, 'w') as f:
            yolo_line = f"{label - 1} 0.5 0.5 1 1\n"
            f.write(yolo_line)


def convert_label_to_yaml(json_path: AnyStr, yaml_path: AnyStr) -> None:
    train = "102flowers/train.txt"
    val = "102flowers/val.txt"
    test = "102flowers/test.txt"

    yaml_header = f"""train: {train}
    val: {val}
    test: {test}

    names:
    """
    with open(json_path) as json_file:
        # Load the JSON data into a Python object
        cat_to_name = json.load(json_file)

    with open(yaml_path, 'w') as file:
        file.write(yaml_header)

        for key, value in cat_to_name.items():
            file.write(f"  {int(key) - 1}: {value}\n")


def predict_from_file(file_path: AnyStr, model_path: AnyStr, output_csv="predictions_yolov5.csv") -> None:
    # Load trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Open the input file containing image paths
    with open(file_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    # Prepare CSV to store predictions
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Path", "Class", "Confidence"])

        # Run inference for each image
        for image_path in image_paths:
            try:
                results = model(image_path)
            except Exception as e:
                print(e)
                continue
            predictions = results.pandas().xyxy[0]  # Convert to DataFrame

            for _, row in predictions.iterrows():
                writer.writerow([image_path, row['name'], row['confidence']])

    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    def run(step: int):
        if step == 0:
            convert_image_to_label(images_path='102flowers/images', labels_path='102flowers/labels',
                                   mat_path='102flowers/imagelabels.mat')
        if step == 1:
            convert_label_to_yaml(json_path='102flowers/cat_to_name.json', yaml_path='102flowers/data.yaml')
        if step == 2:
            command_dict = {
                "freeze": 10,
                "cache": True,
                "epochs": 60,
                "data": "102flowers/data.yaml",
                "weights": "yolov5/yolov5s.pt",
                "batch": 16,
                "patience": 5,
                "project": "runs_flowers",
                "name": "train",
                "device": "0",
                "imgsz": 512,
                "optimizer": "AdamW",
                "cos_lr": True,
                "label_smoothing": 0.1,
                "multi_scale": True,
                "workers": 8,
                "save_period": 5
            }
            train.run(**command_dict)
        if step == 3:
            predict_from_file(file_path='102flowers/test.txt', model_path=r'runs_flowers/train5/weights/best.pt')


    run(3)


