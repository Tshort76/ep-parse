import pandas as pd
import random
import ep_parse.constants as pc
import ep_parse.core as core
import ep_parse.signal_nav as sn
import plotly.graph_objects as pgo
import ep_parse.plots.common as ppc
import ep_parse.utils as pu
import ep_parse.case_data as cdata

QRS_POINTS = ["Q", "S", "T"]


def sample_ecg_signals(case_id: str, time: str = None) -> pd.DataFrame:
    with core.load_case_signals(case_id) as hdfs:
        if time:
            ecg_signals = sn.lookup_by_time(time, channels=pc.CHANNEL_GROUPS["ECG"], signals_store=hdfs)
        else:
            s = random.randint(0, 60) * 60 * 1000
            ecg_signals = hdfs.select(key="signals/ECG", columns=pc.CHANNEL_GROUPS["ECG"], start=s, stop=s + 8000)
            ecg_signals = ecg_signals.dropna(axis="columns")
    return ecg_signals


def plot_for_QRS_marking(ecg_signals: pd.DataFrame, clicks: list[int]) -> pgo.FigureWidget:
    y_axis = [-2, 2]
    y_offsets = iter(ppc.y_shifts(len(ecg_signals.columns), y_axis))

    ordered_offsets = {c: next(y_offsets) for c in ppc.DEFAULT_CHANNEL_ORDER if c in ecg_signals.columns}
    x_index = list(range(len(ecg_signals)))

    fig = ppc.pretty_layout(
        configs={
            "ui_environment": "jupyter",
            "trace_offsets": ordered_offsets,
            "y_axis": y_axis,
            "title": pu.as_time_str(ecg_signals.index[0], False),
        }
    )

    fig.update_xaxes(showspikes=True, spikecolor="orange", spikethickness=1)

    for channel in ordered_offsets.keys():
        signal = ecg_signals[channel]
        fig.add_trace(
            pgo.Scatter(
                x=x_index,
                y=signal + ordered_offsets[channel],
                name=channel,
                hoverinfo="y",
                mode="lines",
                line={"color": "#e8e8e8", "width": 1},
            )
        )

    def update_point(_, points, __):
        if len(points.xs):
            x = points.point_inds[0]
            ppc.add_annotation(fig, x, label=QRS_POINTS[len(clicks)])
            clicks.append(x)

    for scatter in fig.data:
        scatter.on_click(update_point)

    return fig


def save_qrs_marks(case_id: str, ecg_signals: pd.DataFrame, clicks: list[int]) -> None:
    assert len(clicks) == len(QRS_POINTS), f"Expected a click for each of {QRS_POINTS}"
    Q, S, T = clicks
    pps = pu.points_per_second(ecg_signals.index)
    # note that maxes are indexed such that iloc[0] = Q and iloc[-1] == S
    maxes = ecg_signals.iloc[Q:S].abs().reset_index(drop=True).idxmax().to_dict()
    qrs_meta = {"QS_duration_ms": pu.points_as_ms(S - Q, pps), "ST_duration_ms": pu.points_as_ms(T - S, pps)}
    for ch, idx in maxes.items():
        qrs_meta[ch] = {
            "QR": pu.points_as_ms(idx, pps),  # idx is relative to QS region
            "RS": pu.points_as_ms((S - Q) - idx, pps),  # idx is relative to QS region
            "R_volts": float(round(abs(ecg_signals[ch].iloc[Q + idx]), 5)),
        }
    cdata.update_case_meta(case_id, ["qrs_meta"], qrs_meta)
    print(f"Saved qrs_meta to the {case_id}'s case_meta file")


def build_interval(t: str, clicks: list[str]) -> dict:
    return {
        "type": t,
        "baseline_start_time": clicks[0],
        "baseline_end_time": clicks[1],
        "treatment_start_time": clicks[2],
        "treatment_end_time": clicks[3],
    }


# TODO
def save_rhythm_intervals(case_id: str, clicks: list[str], interval_types: list[str]) -> None:
    assert len(clicks) % 4 == 0, "Number of clicks MUST be a multiple of 4 !!"
    assert int(len(clicks) / 4) == len(interval_types), "You must enter an interval type for each interval denoted!"
    intervals = [build_interval(t, clicks[i * 4 : (i + 1) * 4]) for i, t in enumerate(interval_types)]
    cdata.update_case_meta(case_id, ["rhythm_intervals"], intervals, append=False)
