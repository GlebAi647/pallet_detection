import cv2
import os
from typing import List
from utils.csv_handler import save_to_excel
from utils.logger import get_logger
from typing import Any
from detection_pipeline.dto import Box


def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger = get_logger(__name__)
    logger.info(f"Видео загружено: {frame_count} frames at {frame_rate} FPS")

    return cap, frame_count, frame_rate, frame_width, frame_height


def write_video(
        output_video_path: str,
        frame_rate: int,
        frame_width: int,
        frame_height: int
):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        frame_rate,
        (frame_width, frame_height)
    )

    logger = get_logger(__name__)
    logger.info(f"Результирующее видео сохранено: {output_video_path}")

    return out


def save_event_frame(
        events_dir: str,
        frame_idx: int,
        label: str,
        frame
):
    os.makedirs(events_dir,exist_ok=True)
    cv2.imwrite(
        os.path.join(events_dir, f"event_{frame_idx}_{label}.jpg"),
        frame
    )
    logger = get_logger(__name__)
    logger.info(f"Сохраненный фрейм события для {label}: event_{frame_idx}_{label}.jpg")


def save_results(
        data: List[List[Any]],
        output_excel_path: str
):
    save_to_excel(data, output_excel_path)

    logger = get_logger(__name__)
    logger.info(f"Результаты сохранены в Excel: {output_excel_path}")
