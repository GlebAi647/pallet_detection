from detection_pipeline.dto import Box
from detection_pipeline.constants import class_names
from detection_pipeline.video_handler import save_event_frame
from utils.logger import get_logger

logger = get_logger(__name__)


def is_pallet_covered(
        present_box: Box,
        pallet_box: Box) -> bool:
    return (
            present_box.left <= pallet_box.left and
            present_box.top <= pallet_box.top and
            present_box.right >= pallet_box.right and
            present_box.bottom >= pallet_box.bottom
    )


def check_coverage(
        current_objects: dict) -> bool:
    if not current_objects.get('present'):
        return False
    present_box = current_objects['present'][0]
    pallets = current_objects.get('pallet', [])
    return all(is_pallet_covered(present_box, p) for p in pallets)


def process_events(
        frame_idx: int,
        current_objects: dict,
        events_dir: str,
        frame
):
    if len(current_objects.get("human", [])) > 0:
        save_event_frame(
            events_dir,
            frame_idx,
            "human",
            frame.copy())
    if len(current_objects.get("pallet", [])) > 0:
        save_event_frame(
            events_dir,
            frame_idx,
            "pallet",
            frame.copy()
        )
