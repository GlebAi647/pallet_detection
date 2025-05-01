import os
import sys
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from detection_pipeline.constants import class_names
from utils.logger import get_logger
from config.configuration import CONFIG

logger = get_logger(__name__)


def setup_analytics_directory():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(CONFIG["events_dir"], "analitics_report_inference", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Аналитика сохранена в {base_dir}")
    return base_dir


def load_results_from_excel(input_excel_path: str) -> pd.DataFrame:
    if not os.path.exists(input_excel_path):
        raise FileNotFoundError(f"Файл {input_excel_path} не найден.")

    df = pd.read_excel(input_excel_path)
    logger.info(f"Данные загружены из {input_excel_path}")
    return df


def analyze_confidence_by_class(df: pd.DataFrame) -> Dict[str, Dict]:
    results = {}
    for cls_name in class_names.values():
        filtered_df = df[df['Class'] == cls_name]
        if not filtered_df.empty:
            results[cls_name] = {
                'count': len(filtered_df),
                'mean': filtered_df['Confidence'].mean(),
                'std': filtered_df['Confidence'].std(),
                'min': filtered_df['Confidence'].min(),
                'max': filtered_df['Confidence'].max()
            }
        else:
            results[cls_name] = {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            }
    return results


def plot_confidence_distribution(
        df: pd.DataFrame,
        output_dir: str,
        file_prefix: str = "confidence_distribution"
):
    classes = df['Class'].unique()
    for cls_name in classes:
        filtered_df = df[df['Class'] == cls_name]
        if not filtered_df.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(filtered_df['Confidence'], bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Распределение Confidence для класса "{cls_name}"')
            plt.xlabel('Confidence')
            plt.ylabel('Частота')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{file_prefix}_{cls_name}.png"))
            plt.close()
            logger.info(f"График сохранён: {file_prefix}_{cls_name}.png")


def plot_min_max_confidence_per_frame(
        df: pd.DataFrame,
        output_dir: str,
        file_prefix: str = "min_max_confidence_per_frame"
):
    grouped = df.groupby("Frame Number")["Confidence"].agg(['min', 'max']).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(grouped["Frame Number"], grouped["min"], label="Минимальная уверенность", color='red')
    plt.plot(grouped["Frame Number"], grouped["max"], label="Максимальная уверенность", color='blue')
    plt.title("Минимальная и максимальная уверенность на каждом кадре")
    plt.xlabel("Номер кадра")
    plt.ylabel("Уверенность")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}.png"))
    plt.close()
    logger.info(f"График сохранён: {file_prefix}.png")


def save_summary_to_text_file(
        analysis_results: Dict[str, Dict],
        output_dir: str,
        filename: str = "summary.txt"
):
    summary_path = os.path.join(output_dir, filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Анализ уверенности (confidence) ===\n\n")
        for cls_name, stats in analysis_results.items():
            f.write(f"Класс: {cls_name}\n")
            f.write(f"  Объектов: {stats['count']}\n")
            f.write(f"  Средняя уверенность: {stats['mean']:.4f}\n" if stats[
                                                                           'mean'] is not None else "Средняя уверенность: N/A\n")
            f.write(f"  Стандартное отклонение: {stats['std']:.4f}\n" if stats[
                                                                             'std'] is not None else "Стандартное отклонение: N/A\n")
            f.write(f"  Минимум: {stats['min']:.4f}\n" if stats['min'] is not None else "  Минимум: N/A\n")
            f.write(f"  Максимум: {stats['max']:.4f}\n" if stats['max'] is not None else "  Максимум: N/A\n")
            f.write("\n")
    logger.info(f"Статистика сохранена в {summary_path}")


def run_analytics(
        input_excel_path: str,
        output_dir: str,
        file_prefix: str = "inference"
):
    df = load_results_from_excel(input_excel_path)
    analysis_results = analyze_confidence_by_class(df)
    save_summary_to_text_file(analysis_results, output_dir)
    plot_confidence_distribution(df, output_dir, file_prefix)
    plot_min_max_confidence_per_frame(df, output_dir, file_prefix)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root)
    input_excel_path = CONFIG["output_excel_path"]
    analytics_output_dir = setup_analytics_directory()
    run_analytics(input_excel_path, analytics_output_dir)
