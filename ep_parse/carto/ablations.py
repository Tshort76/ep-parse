import os
from datetime import datetime
import pandas as pd

import ep_parse.common as pic


def parse_RF_tags(export_dir: str, time_0: datetime) -> list[dict]:
    # Get the coordinates of each ablation site
    filepath = os.path.join(export_dir, "VisiTagExport", "Sites.txt")
    sites = pd.read_csv(filepath, sep="\\s+")

    # Get the timestamp of each ablation site (SiteIndex)
    filepath = os.path.join(export_dir, "VisiTagExport", "AblationSites.txt")
    abl_sites = pd.read_csv(filepath, sep="\\s+")

    # Join the timestamp and coordinates
    sites_df = sites[["X", "Y", "Z", "DurationTime", "SiteIndex"]].set_index("SiteIndex")
    rf_data = abl_sites.set_index("SiteIndex").join(sites_df)

    # Create tags with time and coordinates, using T0 and time offsets for times
    rf_tags = []
    for _, row in rf_data.sort_values(by="FirstPosTimeStamp").iterrows():
        stime = pic.offset_time_str(time_0, row["FirstPosTimeStamp"])
        etime = pic.offset_time_str(time_0, row["LastPosTimeStamp"])
        rf_tags.append(pic.as_RF_tag(len(rf_tags) + 1, list(row[["X", "Y", "Z"]].values), stime, etime))

    return rf_tags
