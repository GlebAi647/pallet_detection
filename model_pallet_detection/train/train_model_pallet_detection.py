import torch
from ultralytics import YOLO


def train_model(model_path, dataset_path, imgsz, batch, epochs, workers, device):
    model = YOLO(model_path)

    train_params = {
        'data': dataset_path,
        'imgsz': imgsz,
        'batch': batch,
        'epochs': epochs,
        'workers': workers,
        'device': device if device >= 0 else 'cpu'
    }

    model.train(**train_params)


def main():
    model_path = 'yolov8n.pt'
    dataset_path = '/dataset/'
    imgsz = 1280
    batch = 8
    epochs = 200
    workers = 4
    device = 0

    train_model(model_path, dataset_path, imgsz, batch, epochs, workers, device)


if __name__ == '__main__':
    main()
