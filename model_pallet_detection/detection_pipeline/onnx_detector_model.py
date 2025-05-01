import cv2
import datetime
import logging
import numpy as np
import onnxruntime
from typing import Tuple, List, Dict, Any
from detection_pipeline.dto import Box
from detection_pipeline.constants import class_names
from utils.logger import get_logger

logger = get_logger(__name__)


def letter_box_images(
        images: List[np.ndarray],
        dst_shape: Tuple[int, int],
        back_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    src_shape = images[0].shape[:2][::-1]
    scale = min(
        dst_shape[0] / src_shape[0],
        dst_shape[1] / src_shape[1]
    )
    new_shape = (
        int(src_shape[0] * scale),
        int(src_shape[1] * scale)
    )
    processed_images = []
    for image in images:
        resized_image = cv2.resize(
            image,
            new_shape,
            interpolation=cv2.INTER_LINEAR
        )
        top = (dst_shape[1] - new_shape[1]) // 2
        bottom = dst_shape[1] - new_shape[1] - top
        left = (dst_shape[0] - new_shape[0]) // 2
        right = dst_shape[0] - new_shape[0] - left
        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=back_color,
        )
        processed_images.append(padded_image)
    pad_info = {
        'scale': scale,
        'top_pad': top,
        'bottom_pad': bottom,
        'left_pad': left,
        'right_pad': right,
    }
    return processed_images, pad_info


class ONNX_Detector:
    def __init__(self, path):
        self.initialize_model(path)
        logger.info('DEVICE TYPE IS [%s]', onnxruntime.get_device())

    def __call__(self,
                 batch_images,
                 batch_size=1
                 ):
        return self.detect_objects(
            batch_images,
            batch_size
        )

    def initialize_model(
            self,
            path
    ):
        self.session = onnxruntime.InferenceSession(
            path, providers=['CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def detect_objects(
            self,
            batch_images,
            batch_size=1
    ):
        start_detect_time = datetime.datetime.now()
        input_tensor, pad_info = self.prepare_input(
            batch_images, batch_size)
        prepare_time = datetime.datetime.now()
        logger.info(
            'Inferer prepare time: %s' % str(prepare_time - start_detect_time)
        )
        outputs = self.inference(input_tensor)
        inference_time = datetime.datetime.now()
        logger.info(
            'Inferer inference time: %s' % str(inference_time - prepare_time)
        )
        boxes = self.process_output(
            outputs,
            pad_info,
            batch_images[0].shape[:2]
        )
        postprocess_time = datetime.datetime.now()
        logger.info(
            'Inferer postprocess time: %s'
            % str(postprocess_time - inference_time)
        )
        return boxes

    def prepare_input(self,
                      batch_images,
                      batch_size
                      ):
        resized_batch, pad_info = letter_box_images(
            batch_images,
            (self.input_width, self.input_height),
        )
        tensors_image_list = []
        for image in resized_batch:
            input_img = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
            input_img = input_img.transpose(2, 0, 1)
            tensors_image_list.append(input_img)
        input_tensor = (np.stack
                        (tensors_image_list,
                         axis=0).astype(np.float32)
                        )
        input_tensor = input_tensor / 255.0
        return input_tensor, pad_info

    def inference(self, input_tensor):
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_tensor}
        )[0]
        return outputs

    def process_output(
            self,
            outputs,
            pad_info,
            original_shape
    ):
        boxes = []
        scale = pad_info['scale']
        left_pad = pad_info['left_pad']
        top_pad = pad_info['top_pad']
        original_height, original_width = original_shape
        for box in outputs[0]:
            x1, y1, x2, y2, score, class_id = box[:6].astype(float)
            if score > 0.5:
                x1 = int((x1 - left_pad) / scale)
                y1 = int((y1 - top_pad) / scale)
                x2 = int((x2 - left_pad) / scale)
                y2 = int((y2 - top_pad) / scale)
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                x2 = max(0, min(x2, original_width))
                y2 = max(0, min(y2, original_height))

                top = y1
                left = x1
                bottom = y2
                right = x2

                boxes.append(
                    Box(
                        top=top,
                        left=left,
                        bottom=bottom,
                        right=right,
                        score=score,
                        name=class_names[int(class_id)],
                    )
                )
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [input.name for input in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [output.name for output in model_outputs]
