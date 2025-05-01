from ultralytics import YOLO


def convert_model(
        model_path: str,
        result_name: str,
        image_size: int = 1280,
        opset: int = 12,
        dynamic: bool = False,
):
    model = YOLO(model_path)

    model.export(
        format="onnx",
        nms=True,
        imgsz=image_size,
        opset=opset,
        dynamic=dynamic,
        simplify=True,
    )


if __name__ == "__main__":
    MODEL_PATH = "C:/Users/SuperPC/PycharmProjects/PythonProject/model_pallet_detection/exploration/pallet_detection/experiment_1/weights/model_pallet_29.04.25.pt"
    OUTPUT_PATH = "C:/Users/SuperPC/PycharmProjects/PythonProject/model_pallet_detection/exploration/pallet_detection/experiment_1/weights/model_pallet_29.04.25.onnx"
    IMAGE_SIZE = 1280
    OPSET_VERSION = 12
    DYNAMIC_BATCH = False

    convert_model(
        model_path=MODEL_PATH,
        result_name=OUTPUT_PATH,
        image_size=IMAGE_SIZE,
        opset=OPSET_VERSION,
        dynamic=DYNAMIC_BATCH,
    )
