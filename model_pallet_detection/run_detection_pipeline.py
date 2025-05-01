import os
import datetime
import cv2
import numpy as np
from detection_pipeline.dto import Box
from detection_pipeline.onnx_detector_model import ONNX_Detector
from detection_pipeline.video_handler import read_video, write_video, save_event_frame, save_results
from detection_pipeline.event_logic import check_coverage
from detection_pipeline.visualizer import draw_boxes, add_status_text
from detection_pipeline.constants import class_names, class_colors
from utils.logger import get_logger
from utils.csv_handler import save_to_excel
from config.configuration import CONFIG

logger = get_logger(__name__)


def Model_CV():
    model_path = CONFIG["model_path"]
    video_path = CONFIG["video_path"]
    output_excel_path = CONFIG["output_excel_path"]
    output_video_path = CONFIG["output_video_path"]
    events_dir = CONFIG["events_dir"]

    detector = ONNX_Detector(model_path)
    cap, frame_count, frame_rate, frame_width, frame_height = read_video(video_path)
    out = write_video(output_video_path, frame_rate, frame_width, frame_height)

    data = []
    previous_coverage_status = False
    total_covered_events = 0
    total_processing_time = 0
    total_confidence = 0
    total_detections = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_frame_time = datetime.datetime.now()

        results = detector([frame])

        current_objects = {
            'human': [],
            'pallet': [],
            'present': []
        }
        for box in results:
            current_objects[box.name].append(box)

        current_coverage_status = check_coverage(current_objects)

        if current_coverage_status and not previous_coverage_status:
            total_covered_events += 1
            save_event_frame(events_dir, frame_idx, "cover", frame.copy())

        draw_boxes(frame, results)
        status_text = "The pallets are COVERED with a dome!" if current_coverage_status else "Pallets visible"
        add_status_text(frame, status_text, frame_width)

        out.write(frame)

        for box in results:
            data.append([
                frame_idx,
                box.name,
                box.score,
                box.left,
                box.top,
                box.right,
                box.bottom])

        if len(current_objects['human']) > 0:
            save_event_frame(events_dir, frame_idx, "human", frame.copy())
        if len(current_objects['pallet']) > 0:
            save_event_frame(events_dir, frame_idx, "pallet", frame.copy())

        end_frame_time = datetime.datetime.now()
        frame_processing_time = (end_frame_time - start_frame_time).total_seconds()
        total_processing_time += frame_processing_time

        if len(results) > 0:
            total_confidence += sum(box.score for box in results)
            total_detections += len(results)

        frame_idx += 1

    cap.release()
    out.release()
    save_results(data, output_excel_path)

    avg_fps = frame_idx / total_processing_time if total_processing_time > 0 else 0
    avg_confidence = total_confidence / total_detections if total_detections > 0 else 0

    print(f"Результаты сохранены в {output_excel_path}")
    print(f"Видео с результатами сохранено в {output_video_path}")
    print(f"Всего событий покрытия: {total_covered_events}")
    print(f"Средняя скорость обработки: {avg_fps:.2f} FPS")
    print(f"Средняя точность предсказаний: {avg_confidence:.2f}")


if __name__ == "__main__":
    Model_CV()
