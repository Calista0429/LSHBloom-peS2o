from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

SERIES = {
    "Raw": {
        "color": "#4C78A8",
        "tokens": [0, 4, 8, 12, 16, 20, 24, 28, 31.41],
        "hours": [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.71, 0.80],
        "ppl": [18.5, 13.4, 10.6, 9.1, 8.15, 7.55, 7.12, 6.84, 6.70],
        "acc": [48.0, 55.0, 61.5, 66.7, 70.8, 74.0, 76.5, 78.5, 79.8],
    },
    "MinHashLSH": {
        "color": "#F58518",
        "tokens": [0, 4, 8, 12, 16, 20, 24, 28, 28.81],
        "hours": [0, 0.095, 0.19, 0.285, 0.38, 0.475, 0.57, 0.68, 0.70],
        "ppl": [18.5, 13.1, 10.2, 8.70, 7.75, 7.22, 6.83, 6.60, 6.54],
        "acc": [48.0, 56.0, 62.6, 68.0, 72.1, 75.0, 77.5, 79.4, 80.2],
    },
    "LSHBloom": {
        "color": "#54A24B",
        "tokens": [0, 4, 8, 12, 16, 20, 24, 28, 28.75],
        "hours": [0, 0.093, 0.186, 0.279, 0.372, 0.465, 0.558, 0.65, 0.67],
        "ppl": [18.5, 13.0, 10.05, 8.55, 7.62, 7.10, 6.72, 6.51, 6.47],
        "acc": [48.0, 56.3, 63.0, 68.5, 72.7, 75.6, 78.1, 80.0, 80.7],
    },
}


def element(parent, name, attributes=None, text=None):
    child = ET.SubElement(parent, f"{{{SVG_NS}}}{name}", attributes or {})
    child.text = text
    return child


def scale(value, domain, output):
    domain_start, domain_end = domain
    output_start, output_end = output
    return output_start + (value - domain_start) * (output_end - output_start) / (
        domain_end - domain_start
    )


def draw_panel(root, x0, y0, width, height, title, x_key, y_key, x_label, y_label):
    margin = {"left": 72, "right": 24, "top": 46, "bottom": 58}
    plot_x0 = x0 + margin["left"]
    plot_y0 = y0 + margin["top"]
    plot_x1 = x0 + width - margin["right"]
    plot_y1 = y0 + height - margin["bottom"]
    all_x = [value for series in SERIES.values() for value in series[x_key]]
    all_y = [value for series in SERIES.values() for value in series[y_key]]
    x_domain = (0.0, max(all_x) * 1.03)
    y_padding = (max(all_y) - min(all_y)) * 0.08
    y_domain = (min(all_y) - y_padding, max(all_y) + y_padding)

    element(
        root,
        "text",
        {
            "x": str(x0 + width / 2),
            "y": str(y0 + 24),
            "class": "panel-title",
            "text-anchor": "middle",
        },
        title,
    )
    element(
        root,
        "rect",
        {
            "x": str(plot_x0),
            "y": str(plot_y0),
            "width": str(plot_x1 - plot_x0),
            "height": str(plot_y1 - plot_y0),
            "class": "frame",
        },
    )

    for index in range(6):
        fraction = index / 5
        x_value = x_domain[0] + fraction * (x_domain[1] - x_domain[0])
        x_position = scale(x_value, x_domain, (plot_x0, plot_x1))
        element(
            root,
            "line",
            {
                "x1": str(x_position),
                "x2": str(x_position),
                "y1": str(plot_y0),
                "y2": str(plot_y1),
                "class": "grid",
            },
        )
        element(
            root,
            "text",
            {
                "x": str(x_position),
                "y": str(plot_y1 + 20),
                "class": "tick",
                "text-anchor": "middle",
            },
            f"{x_value:.1f}",
        )

        y_value = y_domain[0] + fraction * (y_domain[1] - y_domain[0])
        y_position = scale(y_value, y_domain, (plot_y1, plot_y0))
        element(
            root,
            "line",
            {
                "x1": str(plot_x0),
                "x2": str(plot_x1),
                "y1": str(y_position),
                "y2": str(y_position),
                "class": "grid",
            },
        )
        y_text = f"{y_value:.1f}"
        element(
            root,
            "text",
            {
                "x": str(plot_x0 - 10),
                "y": str(y_position + 4),
                "class": "tick",
                "text-anchor": "end",
            },
            y_text,
        )

    element(
        root,
        "text",
        {
            "x": str((plot_x0 + plot_x1) / 2),
            "y": str(y0 + height - 10),
            "class": "axis-label",
            "text-anchor": "middle",
        },
        x_label,
    )
    y_axis = element(
        root,
        "text",
        {
            "x": str(x0 + 17),
            "y": str((plot_y0 + plot_y1) / 2),
            "class": "axis-label",
            "text-anchor": "middle",
            "transform": f"rotate(-90 {x0 + 17} {(plot_y0 + plot_y1) / 2})",
        },
        y_label,
    )
    y_axis.set("aria-label", y_label)

    for name, series in SERIES.items():
        points = []
        for x_value, y_value in zip(series[x_key], series[y_key]):
            points.append(
                (
                    scale(x_value, x_domain, (plot_x0, plot_x1)),
                    scale(y_value, y_domain, (plot_y1, plot_y0)),
                )
            )
        path_data = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(points)
        )
        element(
            root, "path", {"d": path_data, "stroke": series["color"], "class": "series"}
        )
        for x, y in points:
            element(
                root,
                "circle",
                {
                    "cx": f"{x:.2f}",
                    "cy": f"{y:.2f}",
                    "r": "3.5",
                    "fill": series["color"],
                },
            )
        end_x, end_y = points[-1]
        diamond = f"M {end_x:.2f} {end_y - 6:.2f} L {end_x + 6:.2f} {end_y:.2f} L {end_x:.2f} {end_y + 6:.2f} L {end_x - 6:.2f} {end_y:.2f} Z"
        element(
            root, "path", {"d": diamond, "fill": series["color"], "class": "endpoint"}
        )


def build_svg():
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {"viewBox": "0 0 1800 930", "width": "1200", "height": "620", "role": "img"},
    )
    element(
        root,
        "title",
        text="Expected deduplication training efficiency curves using simulated data",
    )
    element(
        root,
        "desc",
        text="Four simulated plots compare Raw, MinHashLSH, and LSHBloom by token count and GPU hours.",
    )
    style = element(root, "style")
    style.text = """
      .background { fill: #ffffff; }
      text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #17202a; }
      .main-title { font-size: 25px; font-weight: 600; }
      .subtitle { font-size: 14px; fill: #5d6874; }
      .panel-title { font-size: 16px; font-weight: 600; }
      .axis-label { font-size: 13px; }
      .tick { font-size: 11px; fill: #5d6874; }
      .frame { fill: #ffffff; stroke: #9aa4af; stroke-width: 1; }
      .grid { stroke: #dfe3e8; stroke-width: 1; }
      .series { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
      .endpoint { stroke: #ffffff; stroke-width: 1.5; }
      .legend { font-size: 13px; }
    """
    element(
        root,
        "rect",
        {"x": "0", "y": "0", "width": "1800", "height": "930", "class": "background"},
    )
    element(
        root,
        "text",
        {"x": "900", "y": "34", "class": "main-title", "text-anchor": "middle"},
        "Expected peS2o deduplication efficiency curves",
    )
    element(
        root,
        "text",
        {"x": "900", "y": "58", "class": "subtitle", "text-anchor": "middle"},
        "SIMULATED DATA — diamonds mark the end of one epoch",
    )

    legend_x = 650
    for index, (name, series) in enumerate(SERIES.items()):
        x = legend_x + index * 190
        element(
            root,
            "line",
            {
                "x1": str(x),
                "x2": str(x + 28),
                "y1": "83",
                "y2": "83",
                "stroke": series["color"],
                "stroke-width": "3",
            },
        )
        element(root, "text", {"x": str(x + 37), "y": "88", "class": "legend"}, name)

    draw_panel(
        root,
        200,
        100,
        670,
        390,
        "Validation perplexity vs. training tokens",
        "tokens",
        "ppl",
        "Cumulative training tokens (millions)",
        "Validation perplexity",
    )
    draw_panel(
        root,
        930,
        100,
        670,
        390,
        "SciQ normalized accuracy vs. training tokens",
        "tokens",
        "acc",
        "Cumulative training tokens (millions)",
        "Accuracy (%)",
    )
    draw_panel(
        root,
        200,
        510,
        670,
        390,
        "Validation perplexity vs. GPU hours",
        "hours",
        "ppl",
        "Cumulative training GPU hours",
        "Validation perplexity",
    )
    draw_panel(
        root,
        930,
        510,
        670,
        390,
        "SciQ normalized accuracy vs. GPU hours",
        "hours",
        "acc",
        "Cumulative training GPU hours",
        "Accuracy (%)",
    )
    return ET.ElementTree(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "assets"
        / "expected-efficiency-curves.svg",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_svg().write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)


if __name__ == "__main__":
    main()
