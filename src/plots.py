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

# --- HELPER FOR SEPARATE LEGEND ---

def save_separate_legend(handles, labels, save_dir, filename):
    """Erstellt eine separate PNG-Datei, die nur die Legende enthält."""
    if not handles:
        return

    # Höhe der Figure basierend auf Anzahl der Labels schätzen
    fig_leg = plt.figure(figsize=(4, len(labels) * 0.4 + 0.5))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')

    legend = ax_leg.legend(handles, labels, loc='center', frameon=True)

    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())

    # Padding hinzufügen
    bbox.x0 -= 0.1
    bbox.y0 -= 0.1
    bbox.x1 += 0.1
    bbox.y1 += 0.1

    fig_leg.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches=bbox)
    plt.close(fig_leg)


# --- UTILITY FUNCTIONS ---

def combine_jsons(path: str):
    combined_jsons = {}
    if not os.path.exists(path):
        return {}
    files = os.listdir(path)
    for file in files:
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
        if i + 2 < len(parts) and f"{parts[i]}_{parts[i + 1]}_{parts[i + 2]}" == "replace_logits_cosine":
            method = "replace_logits_cosine"
            method_idx = i + 2
            break
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" in ["replace_logits", "single_sentence", "single_word",
                                                                     "generate_buckets"]:
            method = f"{parts[i]}_{parts[i + 1]}"
            method_idx = i + 1
            break

    if method is None:
        return None, None, None

    trigger_parts = []
    last_trigger_idx = method_idx

    for i in range(method_idx + 1, len(parts)):
        part = parts[i]
        if part.replace('.', '').replace('-', '').replace('e', '').isdigit() and not all(c in '01' for c in part):
            break
        trigger_parts.append(part)
        last_trigger_idx = i

    next_idx_global = last_trigger_idx + 2 + 1
    trigger = " ".join(trigger_parts) if trigger_parts else None
    return method, trigger, next_idx_global


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


# --- PLOT FUNCTIONS WITH SEPARATE LEGENDS ---

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

        fig, ax = plt.subplots(figsize=(10, 7))
        line_df = m_df.copy()
        line_df.loc[line_df['value'] > 100, 'value'] = np.nan
        sns.lineplot(data=line_df, x="x_val", y="value", hue="display_label", marker="s", markersize=8, linewidth=2,
                     ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend(): ax.get_legend().remove()

        color_map = {label: handle.get_color() for handle, label in zip(handles, labels)}
        valid_normal = m_df[m_df['value'] <= 100]['value'].dropna()
        y_min = valid_normal.min() if not valid_normal.empty else 0
        y_max = valid_normal.max() if not valid_normal.empty else 100
        padding = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 5
        ax.set_ylim(y_min - padding, y_max + (padding * 3))
        pin_y, text_y = y_max + padding, y_max + (padding * 1.8)

        for _, row in m_df[m_df['value'] > 100].iterrows():
            m_col = color_map.get(row['display_label'], "red")
            ax.scatter(row['x_val'], pin_y, color=m_col, s=64, marker='s', zorder=5, clip_on=False)
            ax.text(row['x_val'], text_y, f"{row['value']:,.0f}%", color=m_col, fontweight='bold', ha='center',
                    clip_on=False, bbox=dict(facecolor='white', alpha=0.8, edgecolor=m_col))

        ax.set_title(f"Performance Trend: {m_name}", pad=25)
        ax.set_ylabel(f"{m_name} (%)")
        ax.set_xlabel("Sequence Length (Bits)" if m_df['is_bit'].any() else "Poisoning Rate")
        ax.grid(True, linestyle='--', alpha=0.3)

        safe_name = m_name.lower().replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f"{safe_name}_trend.png"), dpi=300, bbox_inches='tight')
        save_separate_legend(handles, labels, save_dir, f"legend_{safe_name}.png")
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
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=subset, x="Metric", y="Value", hue="Configuration", palette="muted", errorbar=None, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend(): ax.get_legend().remove()

        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

        set_size_val = set_sizes.get("Bits" if is_bit_flag else "Text", "?")
        ax.set_title(f"Numbers Comparison (Dataset Size: {set_size_val})", fontsize=14, pad=20)
        ax.set_ylabel("Absolute Numbers")
        ax.set_xlabel(None)

        suffix = 'bits' if is_bit_flag else 'text'
        plt.savefig(os.path.join(save_dir, f"numbers_summary_{suffix}.png"), dpi=300, bbox_inches='tight')
        save_separate_legend(handles, labels, save_dir, f"legend_numbers_{suffix}.png")
        plt.close()


def draw_clean_perplexity_graphs(dict_list: list, save_dir: str):
    data_points = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue
            is_bit = is_bit_sequence_trigger(trigger)
            clean_perp = metrics.get("Average clean perplexity:")
            try:
                f_val = float(clean_perp) if clean_perp is not None else np.nan
            except:
                f_val = np.nan
            data_points.append({
                "label": f"{method} (PR {metrics.get('Poisoning rate', 0)})" if is_bit else method,
                "x_val": len(trigger) if is_bit else metrics.get("Poisoning rate", 0),
                "clean_ppl": f_val,
                "is_bit": is_bit
            })

    df = pd.DataFrame(data_points).sort_values("x_val").dropna(subset=['clean_ppl'])
    if df.empty: return

    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(data=df, x="x_val", y="clean_ppl", hue="label", marker="s", markersize=8, linewidth=2,
                        errorbar=None)

    handles, labels = plot.get_legend_handles_labels()
    if plot.get_legend(): plot.get_legend().remove()

    plt.title("Baseline Quality: Average Clean Perplexity")
    plt.ylabel("Clean Perplexity")
    plt.xlabel("Bit Sequence Length" if df['is_bit'].any() else "Poisoning Rate")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(os.path.join(save_dir, "clean_perplexity_trend.png"), dpi=300, bbox_inches='tight')
    save_separate_legend(handles, labels, save_dir, "legend_clean_perplexity.png")
    plt.close()


def draw_perplexity_ratio_graphs(dict_list: list, save_dir: str):
    data_points = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue
            is_bit = is_bit_sequence_trigger(trigger)
            clean_perp = metrics.get("Average clean perplexity:")
            poison_perp = metrics.get("Average poisoned perplexity:")
            ratio = poison_perp / clean_perp if (clean_perp and poison_perp and clean_perp > 0) else np.nan
            data_points.append({
                "label": f"{method} (PR {metrics.get('Poisoning rate', 0)})" if is_bit else method,
                "x_val": len(trigger) if is_bit else metrics.get("Poisoning rate", 0),
                "ratio": ratio,
                "is_bit": is_bit
            })

    df = pd.DataFrame(data_points).sort_values("x_val")
    if df.empty or df['ratio'].isna().all(): return

    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(data=df, x="x_val", y="ratio", hue="label", marker="s", markersize=8, linewidth=2,
                        errorbar=None)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    handles, labels = plot.get_legend_handles_labels()
    if plot.get_legend(): plot.get_legend().remove()

    plt.title("Perplexity Impact Ratio (Poisoned / Clean)")
    plt.ylabel("Ratio")
    plt.xlabel("Bit Sequence Length" if df['is_bit'].any() else "Poisoning Rate")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(os.path.join(save_dir, "perplexity_ratio_trend.png"), dpi=300, bbox_inches='tight')
    save_separate_legend(handles, labels, save_dir, "legend_perplexity_ratio.png")
    plt.close()


# --- MODIFIED COMBINED / GLOBAL PLOTS ---

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
                all_rows.append(
                    {"Method": method, "Metric": metric_name, "x_val": x_val, "Value": f_val, "is_bit": is_bit})

    df = pd.DataFrame(all_rows)
    for m_name in df['Metric'].unique():
        m_df = df[df['Metric'] == m_name].copy()
        m_df = m_df.groupby(['Method', 'x_val', 'is_bit'], as_index=False)['Value'].mean().sort_values("x_val")

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(data=m_df, x="x_val", y="Value", hue="Method", marker="s", markersize=8, linewidth=2, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend(): ax.get_legend().remove()

        ax.set_title(f"Combined Trend: {m_name}")
        ax.set_ylabel(f"{m_name} (%)")
        ax.set_xlabel("Bit Sequence Length" if m_df['is_bit'].any() else "Poisoning Rate")
        ax.grid(True, linestyle='--', alpha=0.3)

        safe_metric = m_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"combined_{safe_metric}.png"), bbox_inches='tight')
        save_separate_legend(handles, labels, save_dir, f"legend_combined_{safe_metric}.png")
        plt.close()


def draw_global_perplexity_ratio_graphs(dict_list: list, save_dir: str):
    all_rows = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger, _ = extract_method_and_trigger(exp_name)
            if not method or not trigger: continue
            is_bit = is_bit_sequence_trigger(trigger)
            x_val = len(trigger) if is_bit else metrics.get("Poisoning rate", 0)
            clean_perp = metrics.get("Average clean perplexity:")
            poison_perp = metrics.get("Average poisoned perplexity:")
            if clean_perp and poison_perp and clean_perp > 0:
                all_rows.append({"Method": method, "x_val": x_val, "Ratio": poison_perp / clean_perp, "is_bit": is_bit})

    df = pd.DataFrame(all_rows)
    if df.empty: return
    df = df.groupby(['Method', 'x_val', 'is_bit'], as_index=False)['Ratio'].mean().sort_values("x_val")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(data=df, x="x_val", y="Ratio", hue="Method", marker="s", markersize=8, linewidth=2, ax=ax)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend(): ax.get_legend().remove()

    ax.set_title("Global Perplexity Impact Ratio (Average over all Sets)")
    ax.set_ylabel("Impact Ratio (Poisoned / Clean)")
    ax.set_xlabel("Bit Sequence Length" if df['is_bit'].any() else "Poisoning Rate")
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(os.path.join(save_dir, "combined_perplexity_ratio.png"), bbox_inches='tight')
    save_separate_legend(handles, labels, save_dir, "legend_combined_perplexity_ratio.png")
    plt.close()


# --- MAIN EXECUTION LOGIC (Rest remains similar) ---

def draw_global_combined_plots(all_sorted_evals: dict, save_path: str):
    global_dir = os.path.join(save_path, "combined")
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
        draw_global_perplexity_ratio_graphs(bit_data, d)
    if non_bit_data:
        d = os.path.join(global_dir, "word_sentence_triggers")
        os.makedirs(d, exist_ok=True)
        draw_global_averaged_metrics(non_bit_data, d)
        draw_global_perplexity_ratio_graphs(non_bit_data, d)


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
            draw_perplexity_ratio_graphs(bit_data, d)
            draw_clean_perplexity_graphs(bit_data, d)
            draw_percentage_graphs_lines(bit_data, d)
        if non_bit_data:
            d = os.path.join(size_dir, "word_sentence_triggers")
            os.makedirs(d, exist_ok=True)
            draw_numbers_graphs(non_bit_data, d)
            draw_perplexity_ratio_graphs(non_bit_data, d)
            draw_clean_perplexity_graphs(non_bit_data, d)
            draw_percentage_graphs_lines(non_bit_data, d)


def draw_test_analysis(save_path):
    test_dir = os.path.join(save_path, "test")
    os.makedirs(test_dir, exist_ok=True)
    if not os.path.exists(TEST_EVALUATION_PATH): return
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
                    var_val = int(parts[next_idx]); x_label = "Epochs"
                except:
                    continue
            elif "bit_sequences" in file_name:
                var_val = trigger;
                x_label = "Trigger Sequence"
            if var_val is None: continue
            clean_perp = metrics.get("Average clean perplexity:")
            poison_perp = metrics.get("Average poisoned perplexity:")
            rows.append({
                "Method": method, "Variable": var_val, "ASR": metrics.get("ASR", 0),
                "FPR": metrics.get("False Positive Rate:", 0), "FNR": metrics.get("False Negative Rate:", 0),
                "Perplexity Ratio": poison_perp / clean_perp if (
                            clean_perp and poison_perp and clean_perp > 0) else np.nan
            })
        df = pd.DataFrame(rows)
        if df.empty: continue
        if "bit_sequences" not in file_name:
            df["Variable"] = pd.to_numeric(df["Variable"])
        df = df.sort_values("Variable")

        sub_dir = os.path.join(test_dir, file_name.replace(".json", ""))
        os.makedirs(sub_dir, exist_ok=True)

        for metric in ["ASR", "FPR", "FNR", "Perplexity Ratio"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x="Variable", y=metric, hue="Method", marker="s", markersize=8, ax=ax)

            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            if metric == "Perplexity Ratio":
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                plt.ylim(min(0.8, df[metric].min() * 0.9), max(1.5, df[metric].max() * 1.1))
            else:
                plt.ylim(-5, 105)

            plt.title(f"Test Series: {metric} vs {x_label}")
            plt.savefig(os.path.join(sub_dir, f"{metric.lower().replace(' ', '_')}_trend.png"), dpi=300)
            save_separate_legend(handles, labels, sub_dir, f"legend_{metric.lower()}.png")
            plt.close()


if __name__ == '__main__':
    combined = combine_jsons(EVALUATION_PATH)
    if combined:
        sorted_evals = sort_evaluations(combined)
        draw_evaluations(sorted_evals, GRAPH_PATH)
        draw_global_combined_plots(sorted_evals, GRAPH_PATH)
    draw_test_analysis(GRAPH_PATH)