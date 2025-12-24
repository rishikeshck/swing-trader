"""
Compute technical indicators for CANBK minute data and export enriched CSV and SVG plots.
Pure-Python implementation (no numpy/pandas/matplotlib) to avoid environment issues.
Indicators: RSI (14), Bollinger Bands (20, 2σ), MACD (12/26/9), EMAs (8/20/50/100/200).
Outputs:
- outputs/CANBK_NSE_2763265_indicators.csv
- outputs/canbk_indicators.svg
- console preview (head/tail/summary)
"""

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


DATA_FILE = Path("CANBK_NSE_2763265.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "CANBK_NSE_2763265_indicators.csv"
OUTPUT_SVG = OUTPUT_DIR / "canbk_indicators.svg"


def parse_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "date": r["date"],
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["volume"]),
                }
            )
    return rows


def ema_series(values: List[float], span: int) -> List[float]:
    alpha = 2 / (span + 1)
    out: List[float] = []
    ema_val = values[0]
    out.append(ema_val)
    for v in values[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
        out.append(ema_val)
    return out


def rsi_series(values: List[float], window: int = 14) -> List[float]:
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = gains[1:window + 1]
    avg_loss = losses[1:window + 1]
    if len(avg_gain) < window:
        return [float("nan")] * len(values)
    gain_val = sum(avg_gain) / window
    loss_val = sum(avg_loss) / window
    rsi_list = [float("nan")] * len(values)
    for i in range(window, len(values)):
        gain_val = (gain_val * (window - 1) + gains[i]) / window
        loss_val = (loss_val * (window - 1) + losses[i]) / window
        if loss_val == 0:
            rsi_list[i] = 100.0
        else:
            rs = gain_val / loss_val
            rsi_list[i] = 100 - (100 / (1 + rs))
    return rsi_list


def rolling_mean_std(values: List[float], window: int) -> Tuple[List[float], List[float]]:
    means = [float("nan")] * len(values)
    stds = [float("nan")] * len(values)
    window_vals: List[float] = []
    s = 0.0
    s2 = 0.0
    for i, v in enumerate(values):
        window_vals.append(v)
        s += v
        s2 += v * v
        if len(window_vals) > window:
            old = window_vals.pop(0)
            s -= old
            s2 -= old * old
        if len(window_vals) == window:
            mean = s / window
            var = max(s2 / window - mean * mean, 0.0)
            means[i] = mean
            stds[i] = math.sqrt(var)
    return means, stds


def macd_series(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = ema_series(values, fast)
    ema_slow = ema_series(values, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema_series(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def add_indicators(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    closes = [r["close"] for r in rows]
    rsi = rsi_series(closes, 14)
    bb_mean, bb_std = rolling_mean_std(closes, 20)
    macd_line, macd_signal, macd_hist = macd_series(closes)
    ema_spans = (8, 20, 50, 100, 200)
    ema_map = {span: ema_series(closes, span) for span in ema_spans}

    out_rows: List[Dict[str, float]] = []
    for i, r in enumerate(rows):
        row = dict(r)
        row["rsi14"] = rsi[i]
        row["bb_mid"] = bb_mean[i]
        row["bb_upper"] = bb_mean[i] + 2 * bb_std[i] if not math.isnan(bb_mean[i]) else float("nan")
        row["bb_lower"] = bb_mean[i] - 2 * bb_std[i] if not math.isnan(bb_mean[i]) else float("nan")
        row["macd"] = macd_line[i]
        row["macd_signal"] = macd_signal[i]
        row["macd_hist"] = macd_hist[i]
        for span in ema_spans:
            row[f"ema_{span}"] = ema_map[span][i]
        out_rows.append(row)
    return out_rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "open", "high", "low", "close", "volume",
        "rsi14", "bb_mid", "bb_upper", "bb_lower",
        "macd", "macd_signal", "macd_hist",
        "ema_8", "ema_20", "ema_50", "ema_100", "ema_200",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def normalize(series: List[float], height: float, pad: float = 0.05) -> List[float]:
    vals = [v for v in series if not math.isnan(v) and not math.isinf(v)]
    if not vals:
        return [0.0] * len(series)
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [height / 2.0] * len(series)
    rng = mx - mn
    mn -= rng * pad
    mx += rng * pad
    rng = mx - mn
    return [(v - mn) / rng * height if not math.isnan(v) else float("nan") for v in series]


def build_svg(rows: List[Dict[str, float]], path: Path) -> None:
    width = 1400
    height = 900
    pad_x = 50
    sections = 3
    sec_height = (height - 40) / sections
    x_step = (width - 2 * pad_x) / max(len(rows) - 1, 1)

    closes = [r["close"] for r in rows]
    bb_upper = [r["bb_upper"] for r in rows]
    bb_mid = [r["bb_mid"] for r in rows]
    bb_lower = [r["bb_lower"] for r in rows]
    rsi = [r["rsi14"] for r in rows]
    macd = [r["macd"] for r in rows]
    macd_signal = [r["macd_signal"] for r in rows]
    macd_hist = [r["macd_hist"] for r in rows]
    ema_8 = [r["ema_8"] for r in rows]
    ema_20 = [r["ema_20"] for r in rows]
    ema_50 = [r["ema_50"] for r in rows]
    ema_100 = [r["ema_100"] for r in rows]
    ema_200 = [r["ema_200"] for r in rows]

    price_y = normalize(closes + bb_upper + bb_lower, sec_height)
    bb_u_y = normalize(bb_upper, sec_height)
    bb_m_y = normalize(bb_mid, sec_height)
    bb_l_y = normalize(bb_lower, sec_height)
    ema8_y = normalize(ema_8, sec_height)
    ema20_y = normalize(ema_20, sec_height)
    ema50_y = normalize(ema_50, sec_height)
    ema100_y = normalize(ema_100, sec_height)
    ema200_y = normalize(ema_200, sec_height)

    rsi_y = [sec_height - v * sec_height / 100 for v in rsi]
    macd_all = [v for v in macd + macd_signal + macd_hist if not math.isnan(v)]
    macd_y = normalize(macd, sec_height)
    macd_signal_y = normalize(macd_signal, sec_height)
    macd_hist_y = normalize(macd_hist, sec_height)

    def polyline(points, color, stroke_width=1.2):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points if not math.isnan(y))
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{pts}" />'

    def bars(values, base_y, height_scale, color):
        out = []
        for idx, v in enumerate(values):
            if math.isnan(v):
                continue
            x = pad_x + idx * x_step
            h = v * height_scale
            y = base_y - h if h >= 0 else base_y
            out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{x_step*0.9:.2f}" height="{abs(h):.2f}" fill="{color}" opacity="0.6" />')
        return "\n".join(out)

    price_section_top = 10
    rsi_section_top = price_section_top + sec_height + 10
    macd_section_top = rsi_section_top + sec_height + 10

    x_coords = [pad_x + i * x_step for i in range(len(rows))]
    price_points = list(zip(x_coords, [price_section_top + sec_height - y for y in price_y[: len(rows)]]))
    bb_u_points = list(zip(x_coords, [price_section_top + sec_height - y for y in bb_u_y[: len(rows)]]))
    bb_m_points = list(zip(x_coords, [price_section_top + sec_height - y for y in bb_m_y[: len(rows)]]))
    bb_l_points = list(zip(x_coords, [price_section_top + sec_height - y for y in bb_l_y[: len(rows)]]))
    ema8_points = list(zip(x_coords, [price_section_top + sec_height - y for y in ema8_y[: len(rows)]]))
    ema20_points = list(zip(x_coords, [price_section_top + sec_height - y for y in ema20_y[: len(rows)]]))
    ema50_points = list(zip(x_coords, [price_section_top + sec_height - y for y in ema50_y[: len(rows)]]))
    ema100_points = list(zip(x_coords, [price_section_top + sec_height - y for y in ema100_y[: len(rows)]]))
    ema200_points = list(zip(x_coords, [price_section_top + sec_height - y for y in ema200_y[: len(rows)]]))

    rsi_points = list(zip(x_coords, [rsi_section_top + y for y in rsi_y[: len(rows)]]))
    macd_points = list(zip(x_coords, [macd_section_top + sec_height - y for y in macd_y[: len(rows)]]))
    macd_signal_points = list(zip(x_coords, [macd_section_top + sec_height - y for y in macd_signal_y[: len(rows)]]))
    macd_bars = bars(macd_hist, macd_section_top + sec_height / 2, sec_height / (max(macd_all) - min(macd_all) + 1e-6) if macd_all else 0, "#7f7f7f")

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial;font-size:12px;}</style>',
        # Price
        f'<text x="10" y="{price_section_top + 14}" fill="black">Price with EMAs & Bollinger</text>',
        polyline(price_points, "#000000", 1.1),
        polyline(bb_u_points, "#888888", 0.8),
        polyline(bb_m_points, "#aaaaaa", 0.8),
        polyline(bb_l_points, "#888888", 0.8),
        polyline(ema8_points, "#2ca02c"),
        polyline(ema20_points, "#1f77b4"),
        polyline(ema50_points, "#ff7f0e"),
        polyline(ema100_points, "#8c564b"),
        polyline(ema200_points, "#9467bd"),
        # RSI
        f'<text x="10" y="{rsi_section_top + 14}" fill="black">RSI 14</text>',
        f'<line x1="{pad_x}" y1="{rsi_section_top + sec_height*0.3}" x2="{width-pad_x}" y2="{rsi_section_top + sec_height*0.3}" stroke="#cccccc" stroke-dasharray="4 4" stroke-width="1"/>',
        f'<line x1="{pad_x}" y1="{rsi_section_top + sec_height*0.7}" x2="{width-pad_x}" y2="{rsi_section_top + sec_height*0.7}" stroke="#cccccc" stroke-dasharray="4 4" stroke-width="1"/>',
        polyline(rsi_points, "#d62728"),
        # MACD
        f'<text x="10" y="{macd_section_top + 14}" fill="black">MACD 12/26/9</text>',
        macd_bars,
        polyline(macd_points, "#1f77b4"),
        polyline(macd_signal_points, "#ff7f0e"),
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def summarize(rows: List[Dict[str, float]]) -> None:
    head = rows[:5]
    tail = rows[-5:]
    print("Preview (head):")
    for r in head:
        print(r)
    print("\nPreview (tail):")
    for r in tail:
        print(r)

    def stats(key: str) -> Tuple[float, float, float]:
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        if not vals:
            return (float("nan"), float("nan"), float("nan"))
        return (min(vals), sum(vals) / len(vals), max(vals))

    keys = ["rsi14", "macd", "macd_signal", "macd_hist", "ema_8", "ema_20", "ema_50", "ema_100", "ema_200"]
    print("\nSummary stats (min/mean/max):")
    for k in keys:
        mn, mean, mx = stats(k)
        print(f"{k:10s} min={mn:.4f} mean={mean:.4f} max={mx:.4f}")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_FILE}")

    rows = parse_rows(DATA_FILE)
    rows_ind = add_indicators(rows)
    write_csv(rows_ind, OUTPUT_CSV)
    build_svg(rows_ind, OUTPUT_SVG)
    summarize(rows_ind)
    print(f"\nSaved enriched CSV to {OUTPUT_CSV}")
    print(f"Saved plot (SVG) to {OUTPUT_SVG}")


if __name__ == "__main__":
    main()

