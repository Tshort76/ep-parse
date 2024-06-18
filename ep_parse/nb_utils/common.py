import toolz as tz
import plotly.graph_objs as go
import ep_parse.signal_nav as sn


def lookup_ids(user_configs: dict):
    lookup_type = user_configs["lookup_type"]
    assert lookup_type in {"EVENT", "RF", "TIME", "TAG"}
    return user_configs.get(f"{lookup_type}S")


def fparams_str(fp: dict):
    s = f"mvc-{tz.get_in(['deflections', 'min_v_change'], fp)}_"
    s += f"ms-{tz.get_in(['deflections', 'min_slope'], fp)}_"
    s += f"mv-{tz.get_in(['deflections', 'min_volts'], fp)}_"
    s += f"md-{tz.get_in(['wavelets', 'min_duration'], fp)}"

    return s


def parse_rf_str(rf_str: str) -> list[int]:
    """Flatten a discontinuous integer ranges into a list of integers

    Args:
        rf_str (str): csv of integer ranges (e.g. '1,3,4-7,9')

    Returns:
        list[str]: corresponding list of integers (e.g. [1,3,4,5,6,7,9])
    """
    rfs = []

    for s in rf_str.split(","):
        if "-" in s:
            spl = s.split("-")
            rfs += list(range(int(spl[0]), int(spl[1]) + 1))
        elif s:
            rfs += [int(s)]

    return rfs


def parse_user_times(t_str: str) -> list[str]:
    """Parse a CSV string of time periods, where each time period is a sequence on stepped intervals

    Args:
        t_str (str): <start-time|direction|duration|step> formatted string

    Returns:
        list[str]: sequence of start-times (in 'HH:MM:ss' format) for the intervals

    Examples:
        parse_user_times('12:00:00|after|20|10')  -->  ['12:00:00.000', '12:00:10.000']
        parse_user_times('12:00:00|after|21|10')  -->  ['12:00:00.000', '12:00:10.000', '12:00:20.000']
        parse_user_times('12:00:00|after|11|5')  -->  ['12:00:00.000', '12:00:05.000', '12:00:10.000'']
        parse_user_times('12:00:00|before|20|10') -->  ['11:59:40.000', '11:59:50.000']

    """
    return sorted(tz.concat([sn.parse_time_str(tm) for tm in t_str.split(",")]))


def expand_tag_labels(tag_type: str, intervals: str) -> list[dict]:
    """Lookup tags by specifying a type (e.g. MAP) and interval groups (same format at rfs above)

    Args:
        tag_type (str): The type of the tag.  Should be a prefix for a label from the tags file (e.g. MAP or BOOKMARK)
        intervals (str): The intervals to grab. Example of format is 1,4,5-7.

    Returns:
        list[str]: Labels for the tags that correspond to the specified type and intervals
    """
    return [f"{tag_type} {i}" for i in parse_rf_str(intervals)]


def pretty_df(dataframe, opts: dict = {}):
    color_row = list(
        tz.take(len(dataframe), tz.interleave([["#1f1f1f"] * len(dataframe), ["#474747"] * len(dataframe)]))
    )
    if round_to := opts.get("round_to"):
        vals = [dataframe[i].round(round_to) for i in dataframe.columns]
    else:
        vals = [dataframe[i] for i in dataframe.columns]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(dataframe.columns), align="left"),
                cells=dict(
                    values=vals,
                    fill_color=[color_row * len(dataframe.columns)],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark", margin=dict(l=5, r=5, b=5, t=5), width=opts.get("width"), height=opts.get("height")
    )
    return fig
