import json
from datetime import datetime, timedelta
import plotly.graph_objects as pgo
import toolz as tz
from operator import itemgetter

import ep_parse.case_data as cdata
import ep_parse.utils as pu
import ep_parse.constants as pc


def _draw_vline(fig, x: float) -> pgo.Figure:
    pfig = pgo.Figure(fig) if isinstance(fig, dict) else fig
    pfig.add_vline(
        x=x,
        line_dash="dash",
        line_color="orange",
        annotation={
            "text": str(int(x / 2)),
            "name": "time",
            "xanchor": "right",
        },
    )
    return pfig


def _register_click(fig, clicks_store: list[tuple], trace, points, selector):
    if points.xs:
        clicks_store.append((points.xs[0], points.ys[0]))
        return _draw_vline(fig, points.xs[0])


def _update_tag_file(tag_file: str, tag: dict, idx: int):
    with open(tag_file, "r") as fp:
        _tags = json.load(fp)

    _tags[idx] = tag

    with open(tag_file, "w") as fp:
        json.dump(_tags, fp, indent=2)


def as_cardiolab_stime(tag, carto_offset: int, seconds_before: int):
    carto_time = tag["start_time"]
    if len(carto_time) < 10:
        carto_time = pc.DEFAULT_DATE + " " + carto_time
    return datetime.fromisoformat(carto_time) + timedelta(seconds=(carto_offset - seconds_before))


def persist_tag(tag: dict, tag_file: str, idx: int, clicks_store: list, cardiolab_time):
    if len(clicks_store) != 2:
        print(
            f"Expected 2 clicks, but {len(clicks_store)} were registered.  Trying again to plot the interactive signals and click your start and end points"
        )
        return idx
    xs = [x[0] for x in clicks_store]
    start_time = cardiolab_time + timedelta(milliseconds=int(min(xs) / 2))
    end_time = cardiolab_time + timedelta(milliseconds=int(max(xs) / 2))

    tag["start_time"] = start_time.isoformat()
    tag["end_time"] = end_time.isoformat()

    _update_tag_file(tag_file, tag, idx)

    return idx + 1


def add_constant_time(tag_file: str, predicate, add_seconds: int):
    with open(tag_file, "r") as fp:
        tags = json.load(fp)
    for tag in tags:
        if predicate(tag):
            start = tag["start_time"]
            if len(start) <= 8:
                start = datetime.fromisoformat(f"{pc.DEFAULT_DATE} {tag['start_time']}")
                tag["start_time"] = start.isoformat()
            else:
                start = datetime.fromisoformat(tag["start_time"])

            end = start + timedelta(seconds=add_seconds)
            tag["end_time"] = end.isoformat()

    new_file = tag_file.replace("_tags.json", "_updated_tags.json")
    with open(new_file, "w") as fp:
        json.dump(tags, fp, indent=2)

    return f"Wrote new file {new_file}. Please review it.  If it looks correct, replace the contents of {tag_file} with those of {new_file}"


def update_RF_times_by_bookmark(bookmark_df, tags: list[dict], sync_time=None) -> None:
    "Updates tags in place"
    sync_time = sync_time or datetime.now().isoformat(timespec="seconds")
    for tag in tags:
        if tag["label"].startswith("RF"):
            rf_num = int(tag["label"][2:].strip())
            _row = bookmark_df[bookmark_df["RF"] == rf_num]
            if not _row.empty:
                tag["start_time"] = pu.as_time_str(_row["start_time"].iloc[0])
                tag["end_time"] = pu.as_time_str(_row["end_time"].iloc[0])
                tag["time_synced_at"] = sync_time


def _overlaps(tags: list[dict]) -> list[str]:
    srt = sorted(tags, key=itemgetter("start_time"))
    return [srt[i]["label"] for i in range(1, len(srt)) if srt[i - 1]["end_time"] > srt[i]["start_time"]]


def find_overlapping_tags(case_id: str) -> list[str]:
    grouped = tz.groupby(lambda x: x["label"][0:3], cdata.load_case_tags(case_id))
    return _overlaps(grouped["MAP"]) + _overlaps(grouped["RF "])


def add_seconds_to_tags(case_id: str, seconds_to_add: float) -> None:
    tags = cdata.load_case_tags(case_id)
    for t in tags:
        s = pu.as_datetime(t["start_time"]) + timedelta(seconds=seconds_to_add)
        t["start_time"] = pu.as_time_str(s)
        if t.get("end_time"):
            e = pu.as_datetime(t["end_time"]) + timedelta(seconds=seconds_to_add)
            t["end_time"] = pu.as_time_str(e)

    cdata.write_case_tags(case_id, tags, mode="w")
    print(f"Updated times for {len(tags)} tags")
