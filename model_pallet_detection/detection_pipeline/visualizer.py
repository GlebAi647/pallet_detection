import cv2
from detection_pipeline.dto import Box
from detection_pipeline.constants import class_colors
from utils.logger import get_logger

logger = get_logger(__name__)


def draw_boxes(frame, boxes):
    for box in boxes:
        color = class_colors[box.name]
        cv2.rectangle(
            frame,
            (box.left, box.top),
            (box.right, box.bottom),
            color, 2
        )
        cv2.putText(
            frame,
            f"{box.name} {box.score:.2f}",
            (box.left, box.top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )


def add_status_text(
        frame,
        status_text,
        frame_width
):
    cv2.putText(
        frame,
        status_text,
        (frame_width // 2 - 200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
