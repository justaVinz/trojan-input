import json
import os
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_EVALUATION_PATH = os.path.join(BASE_DIR, "..", "evaluation", "test")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "evaluation")
GRAPH_PATH = os.path.join(EVALUATION_PATH, "..", "graphs")


def combine_jsons(path: str):
    combined_jsons = {}
    if not os.path.exists(path):
        return {}
    files = os.listdir(path)
    for file in files:
        # EXCLUDE TEST DATA
        if (file != "combined_evaluations.json" and
                file.endswith(".json") and
                not file.startswith("test")):
            file_path = os.path.join(path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_jsons.update(data)

    file_name = "combined_evaluations.json"
    file_path = os.path.join(path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_jsons, f, indent=2, ensure_ascii=False)
    return combined_jsons


def sort_evaluations(evaluation_json: dict):
    sorted_by_size = {}
    for key, value in evaluation_json.items():
        parts = key.split("_")
        if len(parts) > 1:
            size = parts[1]
            if size not in sorted_by_size:
                sorted_by_size[size] = []
            sorted_by_size[size].append({key: value})
    return sorted_by_size


def is_bit_sequence_trigger(trigger: str) -> bool:
    return all(c in '01' for c in str(trigger))


def extract_method_and_trigger(exp_name: str):
    parts = exp_name.split("_")[2:]
    method = None
    method_idx = -1

    for i in range(len(parts)):
        if i + 2 < len(parts) and f"{parts[i]}_{parts[i+1]}_{parts[i+2]}" == "replace_logits_cosine":
            method = "replace_logits_cosine"
            method_idx = i + 2
            break
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i+1]}" in ["replace_logits", "single_sentence", "single_word", "generate_buckets"]:
            method = f"{parts[i]}_{parts[i+1]}"
            method_idx = i + 1
            break

    if method is None:
        return None, None, None  # ← 3 Werte

    trigger_parts = []
    last_trigger_idx = method_idx  # Index des letzten Trigger-Teils (in parts)

    for i in range(method_idx + 1, len(parts)):
        part = parts[i]
        if part.replace('.', '').replace('-', '').replace('e', '').isdigit() and not all(c in '01' for c in part):
            break
        trigger_parts.append(part)
        last_trigger_idx = i

    # next_idx zeigt auf den globalen parts-Index NACH dem Trigger
    # (parts entspricht exp_name.split("_")[2:], also +2 für den globalen Index)
    next_idx_global = last_trigger_idx + 2 + 1  # +2 wegen [2:]-Slice, +1 für nächstes Element

    trigger = " ".join(trigger_parts) if trigger_parts else None
    return method, trigger, next_idx_global  # ← 3 Werte


def separate_data_by_trigger_type(dict_list: list):
    bit_data, non_bit_data = [], []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if method and trigger:
                if is_bit_sequence_trigger(trigger):
                    bit_data.append({exp_name: metrics})
                else:
                    non_bit_data.append({exp_name: metrics})
    return bit_data, non_bit_data


# --- PLOT FUNCTIONS ---


def draw_percentage_graphs_lines(dict_list: list, save_dir: str):
    percentage_keys = ["ASR", "False Positive Rate:", "False Negative Rate:"]
    all_rows = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue
            p_rate = metrics.get("Poisoning rate", 0)
            is_bit = is_bit_sequence_trigger(trigger)
            x_val = len(trigger) if is_bit else p_rate
            base_label = f"{method} (PR {p_rate})" if is_bit else method
            for k in percentage_keys:
                val = metrics.get(k)
                try:
                    f_val = float(val)
                except:
                    f_val = np.nan
                all_rows.append(
                    {"display_label": base_label, "metric": k.replace(":", "").strip(), "x_val": x_val, "value": f_val,
                     "is_bit": is_bit})

    df = pd.DataFrame(all_rows)
    for m_name in df['metric'].unique():
        m_df = df[df['metric'] == m_name].copy()
        m_df = m_df.groupby(['display_label', 'x_val', 'is_bit'], as_index=False)['value'].mean().sort_values("x_val")

        for label in m_df['display_label'].unique():
            if m_df[m_df['display_label'] == label]['value'].isna().all():
                m_df.loc[m_df['display_label'] == label, 'display_label'] += " (NaN/Fail)"

        fig, ax = plt.subplots(figsize=(12, 7))
        line_df = m_df.copy()
        # Cap normal values at 100 for visualization
        line_df.loc[line_df['value'] > 100, 'value'] = np.nan
        sns.lineplot(data=line_df, x="x_val", y="value", hue="display_label", marker="s", markersize=8, linewidth=2,
                     ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        color_map = {label: handle.get_color() for handle, label in zip(handles, labels)}

        valid_normal = m_df[m_df['value'] <= 100]['value'].dropna()
        y_min = valid_normal.min() if not valid_normal.empty else 0
        y_max = valid_normal.max() if not valid_normal.empty else 100
        padding = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 5
        ax.set_ylim(y_min - padding, y_max + (padding * 3))
        pin_y, text_y = y_max + padding, y_max + (padding * 1.8)

        # Handle massive outliers by pinning them to the top of the graph
        for _, row in m_df[m_df['value'] > 100].iterrows():
            m_col = color_map.get(row['display_label'], "red")
            ax.scatter(row['x_val'], pin_y, color=m_col, s=64, marker='s', zorder=5, clip_on=False)
            ax.text(row['x_val'], text_y, f"{row['value']:,.0f}%", color=m_col, fontweight='bold', ha='center',
                    clip_on=False, bbox=dict(facecolor='white', alpha=0.8, edgecolor=m_col))

        ax.set_title(f"Performance Trend: {m_name}", pad=25)
        ax.set_ylabel(f"{m_name} (%)")
        ax.set_xlabel("Sequence Length (Bits)" if m_df['is_bit'].any() else "Poisoning Rate")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.savefig(os.path.join(save_dir, f"{m_name.lower().replace(' ', '_')}_trend.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()


def draw_numbers_graphs(dict_list: list, save_dir: str):
    number_keys = ["Times triggered", "Times trigger provided", "Times clean data"]
    data_points, set_sizes = [], {}
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue
            p_rate = metrics.get("Poisoning rate", 0)
            is_bit = is_bit_sequence_trigger(trigger)
            label = f"{method} (PR {p_rate})" if is_bit else method
            set_sizes["Bits" if is_bit else "Text"] = metrics.get("All entries", "Unknown")
            for k in number_keys:
                val = metrics.get(k)
                if val is not None:
                    data_points.append(
                        {"Configuration": label, "Metric": k.replace(":", "").strip(), "Value": float(val),
                         "is_bit": is_bit})

    df = pd.DataFrame(data_points)
    for is_bit_flag in [True, False]:
        subset = df[df['is_bit'] == is_bit_flag]
        if subset.empty: continue
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=subset, x="Metric", y="Value", hue="Configuration", palette="muted", errorbar=None)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

        set_size_val = set_sizes.get("Bits" if is_bit_flag else "Text", "?")
        plt.title(f"Numbers Comparison (Dataset Size: {set_size_val})", fontsize=14, pad=20)
        plt.ylabel("Absolute Numbers")
        ax.set_xlabel(None)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.savefig(os.path.join(save_dir, f"numbers_summary_{'bits' if is_bit_flag else 'text'}.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()


def draw_perplexity_graphs(dict_list: list, save_dir: str):
    data_points = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue

            is_bit = is_bit_sequence_trigger(trigger)
            # Average of Clean and Poisoned Perplexity
            vals = [metrics.get(k) for k in ["Average clean perplexity:", "Average poisoned perplexity:"] if
                    metrics.get(k)]

            data_points.append({
                "label": f"{method} (PR {metrics.get('Poisoning rate', 0)})" if is_bit else method,
                "x_val": len(trigger) if is_bit else metrics.get("Poisoning rate", 0),
                "perplexity": sum(vals) / len(vals) if vals else np.nan,
                "is_bit": is_bit
            })

    df = pd.DataFrame(data_points).sort_values("x_val")
    if df.empty: return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="x_val",
        y="perplexity",
        hue="label",
        marker="s",
        markersize=8,
        linewidth=2,
        errorbar=None
    )

    plt.title("Perplexity Trend")
    plt.ylabel("Average Perplexity")
    plt.xlabel("Bit Sequence Length" if df['is_bit'].any() else "Poisoning Rate")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(save_dir, "perplexity_trend.png"), dpi=300, bbox_inches='tight')
    plt.close()


# --- COMBINED ANALYSIS FUNCTIONS ---


def draw_global_averaged_metrics(dict_list: list, save_dir: str):
    percentage_keys = ["ASR", "False Positive Rate:", "False Negative Rate:"]
    all_rows = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue

            is_bit = is_bit_sequence_trigger(trigger)
            x_val = len(trigger) if is_bit else metrics.get("Poisoning rate", 0)

            for k in percentage_keys:
                val = metrics.get(k)
                metric_name = k.replace(":", "").strip()
                try:
                    f_val = float(val)
                    if f_val > 100 and "ASR" not in k: f_val = np.nan
                except:
                    f_val = np.nan

                all_rows.append({
                    "Method": method,
                    "Metric": metric_name,
                    "x_val": x_val,
                    "Value": f_val,
                    "is_bit": is_bit
                })

    df = pd.DataFrame(all_rows)
    for m_name in df['Metric'].unique():
        m_df = df[df['Metric'] == m_name].copy()
        # Aggregate mean values across all dataset sizes
        m_df = m_df.groupby(['Method', 'x_val', 'is_bit'], as_index=False)['Value'].mean().sort_values("x_val")

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=m_df, x="x_val", y="Value", hue="Method",
            marker="s", markersize=8, linewidth=2, ax=ax
        )

        ax.set_title(f"Combined Trend: {m_name}")
        ax.set_ylabel(f"{m_name} (%)")
        is_bit_plot = m_df['is_bit'].any()
        ax.set_xlabel("Bit Sequence Length" if is_bit_plot else "Poisoning Rate")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        safe_metric = m_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"combined_{safe_metric}.png"), bbox_inches='tight')
        plt.close()


def draw_global_perplexity_graphs(dict_list: list, save_dir: str):
    all_rows = []
    p_keys = ["Average clean perplexity:", "Average poisoned perplexity:"]

    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue

            is_bit = is_bit_sequence_trigger(trigger)
            x_val = len(trigger) if is_bit else metrics.get("Poisoning rate", 0)

            vals = [metrics.get(k) for k in p_keys if metrics.get(k) is not None and metrics.get(k) > 0]
            if vals:
                avg_val = sum(vals) / len(vals)
                all_rows.append({
                    "Method": method, "x_val": x_val,
                    "Value": avg_val, "is_bit": is_bit
                })

    df = pd.DataFrame(all_rows)
    if df.empty: return

    # Aggregate across all dataset sizes
    df = df.groupby(['Method', 'x_val', 'is_bit'], as_index=False)['Value'].mean().sort_values("x_val")

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="x_val", y="Value", hue="Method",
                 marker="s", markersize=8, linewidth=2, errorbar=None)

    plt.title("Combined Perplexity Trend (Average)")
    plt.ylabel("Average Perplexity")
    plt.xlabel("Bit Sequence Length" if df['is_bit'].any() else "Poisoning Rate")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(os.path.join(save_dir, "combined_perplexity.png"), bbox_inches='tight')
    plt.close()


def draw_global_combined_plots(all_sorted_evals: dict, save_path: str):
    global_dir = os.path.join(save_path, "combined_analysis")
    os.makedirs(global_dir, exist_ok=True)
    all_data_list = []
    for size, dict_list in all_sorted_evals.items():
        for exp_dict in dict_list:
            for exp_name, metrics in exp_dict.items():
                m_copy = metrics.copy()
                m_copy['set_size'] = size
                all_data_list.append({exp_name: m_copy})

    bit_data, non_bit_data = separate_data_by_trigger_type(all_data_list)
    if bit_data:
        d = os.path.join(global_dir, "bit_sequences")
        os.makedirs(d, exist_ok=True)
        draw_global_averaged_metrics(bit_data, d)
        draw_global_perplexity_graphs(bit_data, d)
    if non_bit_data:
        d = os.path.join(global_dir, "word_sentence_triggers")
        os.makedirs(d, exist_ok=True)
        draw_global_averaged_metrics(non_bit_data, d)
        draw_global_perplexity_graphs(non_bit_data, d)


def draw_evaluations(sorted_evals: dict, save_path: str = "plots"):
    sizes_sorted = sorted(list(sorted_evals.keys()), key=lambda x: int(x) if x.isdigit() else 0)
    for size in sizes_sorted:
        size_dir = os.path.join(save_path, f"graphs_for_size_{size}")
        os.makedirs(size_dir, exist_ok=True)
        bit_data, non_bit_data = separate_data_by_trigger_type(sorted_evals[size])
        if bit_data:
            d = os.path.join(size_dir, "bit_sequences")
            os.makedirs(d, exist_ok=True)
            draw_numbers_graphs(bit_data, d)
            draw_perplexity_graphs(bit_data, d)
            draw_percentage_graphs_lines(bit_data, d)
        if non_bit_data:
            d = os.path.join(size_dir, "word_sentence_triggers")
            os.makedirs(d, exist_ok=True)
            draw_numbers_graphs(non_bit_data, d)
            draw_perplexity_graphs(non_bit_data, d)
            draw_percentage_graphs_lines(non_bit_data, d)


# --- TEST DATA FUNCTIONS ---


def draw_test_analysis(save_path):
    test_dir = os.path.join(save_path, "test")
    os.makedirs(test_dir, exist_ok=True)

    if not os.path.exists(TEST_EVALUATION_PATH):
        return

    test_files = [f for f in os.listdir(TEST_EVALUATION_PATH) if f.startswith("test") and f.endswith(".json")]

    for file_name in test_files:
        file_path = os.path.join(TEST_EVALUATION_PATH, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for exp_name, metrics in data.items():
            method, trigger, next_idx = extract_method_and_trigger(exp_name)
            if method is None: continue

            parts = exp_name.split("_")
            var_val = None
            x_label = "Unbekannt"

            if "set_sizes" in file_name:
                var_val = int(parts[1])
                x_label = "Dataset Size"
            elif "epochs" in file_name:
                try:
                    var_val = int(parts[next_idx])
                    x_label = "Epochs"
                except (IndexError, ValueError):
                    continue
            elif "bit_sequences" in file_name:
                var_val = trigger
                x_label = "Trigger Sequence"

            if var_val is None: continue

            p_keys = ["Average clean perplexity:", "Average poisoned perplexity:"]
            vals = [metrics.get(k) for k in p_keys if metrics.get(k)]
            avg_perp = sum(vals) / len(vals) if vals else np.nan

            rows.append({
                "Method": method,
                "Variable": var_val,
                "ASR": metrics.get("ASR", 0),
                "FPR": metrics.get("False Positive Rate:", 0),
                "FNR": metrics.get("False Negative Rate:", 0),
                "Perplexity": avg_perp
            })

        if not rows:
            print(f"Keine Daten für {file_name} gefunden.")
            continue

        df = pd.DataFrame(rows)

        if "bit_sequences" in file_name:
            df = df.sort_values(["Method", "Variable"])
        else:
            df["Variable"] = pd.to_numeric(df["Variable"])
            df = df.sort_values("Variable")

        sub_dir = os.path.join(test_dir, file_name.replace(".json", ""))
        os.makedirs(sub_dir, exist_ok=True)

        for metric in ["ASR", "FPR", "FNR", "Perplexity"]:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="Variable", y=metric, hue="Method", marker="s", markersize=8, linewidth=2)

            plt.title(f"Test Series: {metric} vs {x_label}")
            plt.ylabel(f"{metric} (%)" if metric != "Perplexity" else "Avg Perplexity")
            plt.xlabel(x_label)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')

            save_file = os.path.join(sub_dir, f"{metric.lower().replace(' ', '_')}_trend.png")
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    combined = combine_jsons(EVALUATION_PATH)
    if combined:
        sorted_evals = sort_evaluations(combined)
        draw_evaluations(sorted_evals, GRAPH_PATH)
        draw_global_combined_plots(sorted_evals, GRAPH_PATH)
    draw_test_analysis(GRAPH_PATH)