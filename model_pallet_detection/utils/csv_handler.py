import pandas as pd
import os


def save_to_excel(data: list, output_excel_path: str):
    df = pd.DataFrame(data, columns=["Frame Number", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    df.to_excel(output_excel_path, index=False)
