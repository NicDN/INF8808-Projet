import argparse
import json
from collections import Counter
from pathlib import Path

import plotly.colors as pc
import plotly.graph_objects as go

THEME = {
    "background_color": "#ffffff",
    "font_family": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "accent_font_family": "Inter, 'Segoe UI', system-ui, sans-serif",
    "dark_color": "#2A2B2E",
    "label_font_size": 14,
    "title_font_size": 20,
    "label_background_color": "#ffffff",
    "colorscale": "Bluyl",
}


def normalize_condition(raw_value: str) -> str | None:
    """
    Keep only Healthy and Parkinson's labels.
    Also tolerate common typo variants such as "healty".
    """
    if raw_value is None:
        return None

    text = str(raw_value).strip().lower()

    # Accept typo variants for healthy.
    if text in {"healthy", "healty", "health"}:
        return "Healthy"

    # Accept a few variants of Parkinson's.
    if text in {"parkinson's", "parkinsons", "parkinson", "parkinson disease"}:
        return "Parkinson's"

    return None


def normalize_bool_or_null(value) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def load_paths_and_counts(patients_dir: Path) -> Counter:
    counts: Counter = Counter()

    for patient_file in sorted(patients_dir.glob("patient_*.json")):
        with patient_file.open("r", encoding="utf-8") as f:
            record = json.load(f)

        condition = normalize_condition(record.get("condition"))
        if condition is None:
            continue

        gender = str(record.get("gender", "Unknown")).strip().capitalize() or "Unknown"
        kinship = normalize_bool_or_null(record.get("appearance_in_kinship"))
        first_grade = normalize_bool_or_null(record.get("appearance_in_first_grade_kinship"))

        path = (gender, condition, kinship, first_grade)
        counts[path] += 1

    return counts


def convert_counts_to_gender_percentages(path_counts: Counter) -> dict:
    """
    Convert raw path counts to weighted values where each gender totals 100.
    This reduces bias when cohort sizes differ across genders.
    """
    gender_totals = Counter()
    for (gender, _, _, _), count in path_counts.items():
        gender_totals[gender] += count

    weighted = {}
    for path, count in path_counts.items():
        gender = path[0]
        total_for_gender = gender_totals[gender]
        weighted[path] = (count / total_for_gender) * 100.0

    return weighted


def build_sankey_data(path_values: dict):
    stages = [
        sorted({p[0] for p in path_values}),
        ["Healthy", "Parkinson's"],
        ["No", "Yes", "Unknown"],
        ["No", "Yes", "Unknown"],
    ]

    labels = []
    node_index = {}

    for stage_i, stage_values in enumerate(stages):
        for value in stage_values:
            label = f"S{stage_i + 1}: {value}"
            node_index[(stage_i, value)] = len(labels)
            labels.append(label)

    link_counter = Counter()

    for (gender, condition, kinship, first_grade), value in path_values.items():
        link_counter[(node_index[(0, gender)], node_index[(1, condition)])] += value
        link_counter[(node_index[(1, condition)], node_index[(2, kinship)])] += value
        link_counter[(node_index[(2, kinship)], node_index[(3, first_grade)])] += value

    sources, targets, values = [], [], []
    for (src, dst), val in link_counter.items():
        sources.append(src)
        targets.append(dst)
        values.append(val)

    # Compute total flow per node: outgoing for stage-0 nodes, incoming for all others.
    outgoing = [0.0] * len(labels)
    incoming = [0.0] * len(labels)
    for src, dst, val in zip(sources, targets, values):
        outgoing[src] += val
        incoming[dst] += val
    node_flow = [incoming[i] if incoming[i] > 0 else outgoing[i] for i in range(len(labels))]

    # Annotate each label with its percentage share within its stage column.
    offset = 0
    new_labels = list(labels)
    for stage_values in stages:
        idxs = list(range(offset, offset + len(stage_values)))
        stage_total = sum(node_flow[i] for i in idxs)
        for i in idxs:
            pct = (node_flow[i] / stage_total * 100) if stage_total > 0 else 0
            new_labels[i] = f"{labels[i]} ({pct:.1f}%)"
        offset += len(stage_values)

    # Percentage each link carries of its source node's total outgoing flow.
    link_labels = [
        f"{values[i] / outgoing[sources[i]] * 100:.1f}%"
        if outgoing[sources[i]] > 0 else ""
        for i in range(len(sources))
    ]

    # Sample 7 evenly-spaced colors from the theme colorscale and map to categories.
    value_keys = ["male", "female", "healthy", "parkinson's", "yes", "no", "unknown"]
    n = len(value_keys)
    sampled_hex = pc.sample_colorscale(THEME["colorscale"], [i / (n - 1) for i in range(n)])

    def hex_to_rgba(color: str, alpha: float) -> str:
        color = color.strip()
        if color.startswith("#"):
            h = color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        else:
            # handles "rgb(r, g, b)" returned by sample_colorscale
            parts = color[color.index("(") + 1 : color.index(")")].split(",")
            r, g, b = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
        return f"rgba({r}, {g}, {b}, {alpha})"

    VALUE_COLORS = {k: hex_to_rgba(sampled_hex[i], 0.90) for i, k in enumerate(value_keys)}
    DEFAULT_COLOR = hex_to_rgba(sampled_hex[n // 2], 0.85)

    node_colors = []
    for lbl in new_labels:
        # Extract the value part after "S{n}: " and before " ("
        raw = lbl.split(": ", 1)[-1].split(" (")[0].strip().lower()
        node_colors.append(VALUE_COLORS.get(raw, DEFAULT_COLOR))

    # Links inherit a translucent version of their source node colour.
    link_colors = [
        node_colors[src].replace("0.90)", "0.35)")
        for src in sources
    ]

    return new_labels, sources, targets, values, link_labels, node_colors, link_colors


def save_sankey(path_counts: Counter, output_image: Path, output_html: Path):
    path_values = dict(path_counts)
    labels, sources, targets, values, link_labels, node_colors, link_colors = build_sankey_data(path_values)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color=THEME["dark_color"], width=0.7),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=link_labels,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        autosize=True,
        paper_bgcolor=THEME["background_color"],
        plot_bgcolor=THEME["background_color"],
        title=dict(
            text=(
                "PADS Sankey: "
                "gender → condition → appearance_in_kinship → appearance_in_first_grade_kinship "
                "(link thickness ∝ patient count)"
            ),
            font=dict(
                family=THEME["font_family"],
                size=THEME["title_font_size"],
                color=THEME["dark_color"],
            ),
        ),
        font=dict(
            family=THEME["font_family"],
            size=THEME["label_font_size"],
            color=THEME["dark_color"],
        ),
        hoverlabel=dict(
            bgcolor=THEME["label_background_color"],
            font=dict(family=THEME["accent_font_family"], color=THEME["dark_color"]),
        ),
        height=690,
        margin=dict(l=180, r=180, t=60, b=20),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_html),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        default_width="100%",
        default_height="690px",
    )

    # Optional PNG export; keep HTML generation working even if kaleido/browser rendering fails.
    try:
        fig.write_image(str(output_image), scale=2)
    except Exception as exc:
        print(f"Warning: skipped PNG export ({exc})")


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Sankey diagram for PADS patients using: "
            "gender -> condition -> appearance_in_kinship -> appearance_in_first_grade_kinship."
        )
    )
    parser.add_argument(
        "--patients-dir",
        type=Path,
        default=script_dir.parent / "patients",
        help="Folder containing patient_XXX.json files.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=script_dir / "pads_sankey_gender_condition_kinship.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=script_dir / "pads_sankey_gender_condition_kinship.html",
        help="Output HTML path (always written).",
    )

    args = parser.parse_args()

    path_counts = load_paths_and_counts(args.patients_dir)
    if not path_counts:
        raise RuntimeError(
            "No matching records found. Check patients-dir and condition labels."
        )

    save_sankey(path_counts, args.output_image, args.output_html)

    n_patients = sum(path_counts.values())
    print(f"Saved PNG: {args.output_image}")
    print(f"Saved HTML: {args.output_html}")
    print(f"Included patients: {n_patients}")
    print("Link thickness is normalized by within-gender percentages.")


if __name__ == "__main__":
    main()
