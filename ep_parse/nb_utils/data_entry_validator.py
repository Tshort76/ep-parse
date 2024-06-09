import os
import numpy as np
import pandas as pd
import toolz as tz
import ep_parse.utils as pu
import ep_parse.case_data as cdata


def _nonempty_date(f) -> str:
    "returns date string or None"
    return f if f and f is not np.nan else None


def _invalid_dates_in_table(df: pd.DataFrame, date_fields: list[str] = ["start_time", "time", "end_time"]) -> list[str]:
    "Return list of lines with bad dates"
    date_cols = df.columns.intersection(date_fields)
    problems = []
    for _, row in df.iterrows():
        for col in date_cols:
            try:
                if dt := _nonempty_date(row[col]):
                    pu.as_datetime(dt)
            except ValueError as err:
                line_summary = ",".join(map(str, row.values))
                problems.append(f"{err} : {line_summary}")
    return problems


def _crUI_ann_files():
    configs = cdata.load_ui_configs("case_review", ["core"])
    return tz.get_in(["arrhythmia_state_plot", "annotations", "csv_files"], configs, [])


def check_dates() -> list[str]:
    issues = []
    files_to_check = cdata.core_annotation_csvs()
    files_to_check.append(os.path.join(cdata.annotations_directory(), "sigUI_annotations.csv"))
    files_to_check += _crUI_ann_files()

    for f in files_to_check:
        df = pd.read_csv(f)
        file_issues = _invalid_dates_in_table(df)
        issues += [f"In file: {f.split(os.path.sep)[-1]}, {iss}" for iss in file_issues]
    return issues
