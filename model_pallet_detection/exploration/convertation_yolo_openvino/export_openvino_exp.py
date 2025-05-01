from ultralytics import YOLO

model = YOLO("C:/Users/SuperPC/PycharmProjects/PythonProject/model_pallet_detection/exploration/pallet_detection/experiment_1/weights/model_pallet_29.04.25.pt")

model.export(
    format="openvino",
    imgsz="1280",
    int8=True,
    nms=True,
    batch=1,
    data="C:/Users/SuperPC/PycharmProjects/PythonProject/model_pallet_detection/train/data_train/pallet_dataset/data.yaml",
    device="cpu")

