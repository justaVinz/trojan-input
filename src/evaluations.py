import json
import os
import math
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def combine_jsons(path: str):
    combined_jsons = {}
    files = os.listdir(path)
    for file in files:
        if file != "all_evaluations.json" and file.endswith(".json"):
            file_path = os.path.join(path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_jsons.update(data)

    file_name = "all_evaluations.json"
    file_path = os.path.join(path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_jsons, f, indent=2, ensure_ascii=False)

    return combined_jsons


def sort_evaluations(evaluation_json: dict):
    sorted_by_size = {}
    for key, value in evaluation_json.items():
        size = key.split("_")[1]
        if size not in sorted_by_size:
            sorted_by_size[size] = []
        sorted_by_size[size].append({key: value})
    return sorted_by_size


def is_bit_sequence_trigger(trigger: str) -> bool:
    return all(c in '01' for c in trigger)


def extract_method_and_trigger(exp_name: str):
    # remove model and set size
    parts = exp_name.split("_")[2:]

    method_keywords = ["replace_logits_cosine", "replace_logits", "generate_buckets", "single_sentence", "single_word"]
    method = None
    method_idx = -1

    for i in range(len(parts)):
        # replace_logits_cosine
        if i + 2 < len(parts) and f"{parts[i]}_{parts[i + 1]}_{parts[i + 2]}" == "replace_logits_cosine":
            method = "replace_logits_cosine"
            method_idx = i + 2
            break
        # replace_logits
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" == "replace_logits":
            method = "replace_logits"
            method_idx = i + 1
            break
        # single_sentence
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" == "single_sentence":
            method = "single_sentence"
            method_idx = i + 1
            break
        # single_word
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" == "single_word":
            method = "single_word"
            method_idx = i + 1
            break
        # generate_buckets
        elif i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" == "generate_buckets":
            method = "generate_buckets"
            method_idx = i + 1
            break

    if method is None:
        return None, None

    trigger_parts = []
    for i in range(method_idx + 1, len(parts)):
        part = parts[i]
        if part.replace('.', '').replace('-', '').replace('e', '').isdigit() and not all(c in '01' for c in part):
            break
        trigger_parts.append(part)

    if trigger_parts:
        trigger = " ".join(trigger_parts)
        return method, trigger

    return method, None


def separate_data_by_trigger_type(dict_list: list):
    bit_data = []
    non_bit_data = []

    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger = extract_method_and_trigger(exp_name)

            if method and trigger:
                if is_bit_sequence_trigger(trigger):
                    bit_data.append({exp_name: metrics})
                else:
                    non_bit_data.append({exp_name: metrics})

    return bit_data, non_bit_data


def draw_evaluations(sorted_evals: dict, save_path: str = "plots"):
    sizes_sorted = list(sorted_evals.keys())
    sizes_sorted.sort()

    for size in sizes_sorted:
        size_dir = os.path.join(save_path, f"graphs_for_size_{size}")
        os.makedirs(size_dir, exist_ok=True)

        dict_list = sorted_evals[size]

        bit_data, non_bit_data = separate_data_by_trigger_type(dict_list)

        if bit_data:
            bit_dir = os.path.join(size_dir, "bit_sequences")
            os.makedirs(bit_dir, exist_ok=True)
            draw_perplexity_graphs(bit_data, bit_dir)
            draw_number_graphs(bit_data, bit_dir)
            draw_percentage_graphs_lines(bit_data, bit_dir)

        if non_bit_data:
            non_bit_dir = os.path.join(size_dir, "word_sentence_triggers")
            os.makedirs(non_bit_dir, exist_ok=True)
            draw_perplexity_graphs(non_bit_data, non_bit_dir)
            draw_number_graphs(non_bit_data, non_bit_dir)
            draw_percentage_graphs_lines(non_bit_data, non_bit_dir)


def draw_perplexity_graphs(dict_list: list, save_dir: str):
    data_points = []

    sample_name = list(dict_list[0].keys())[0] if dict_list else ""
    _, sample_trigger = extract_method_and_trigger(sample_name)
    has_non_bit_triggers = sample_trigger and not is_bit_sequence_trigger(sample_trigger)

    # collect data from dictionaries
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger = extract_method_and_trigger(exp_name)

            if not method or not trigger:
                continue

            perplexity = metrics.get("Average perplexity:", math.nan)
            poisoning_rate = metrics.get("Poisoning rate", math.nan)

            if perplexity is not None and not math.isnan(perplexity):
                if has_non_bit_triggers:
                    data_points.append({
                        'method': method,
                        'trigger': trigger,
                        'perplexity': perplexity,
                        'poisoning_rate': poisoning_rate
                    })
                else:
                    bit_len = len(trigger)
                    data_points.append({
                        'method': method,
                        'bit_length': bit_len,
                        'perplexity': perplexity,
                        'poisoning_rate': poisoning_rate
                    })

    if not data_points:
        return

    # combine similar results and average perplexity
    df = pd.DataFrame(data_points)

    if has_non_bit_triggers:
        group_cols = ['method', 'poisoning_rate', 'trigger']
        x_col = 'trigger'
        x_label = "Trigger Type"
    else:
        group_cols = ['method', 'poisoning_rate', 'bit_length']
        x_col = 'bit_length'
        x_label = "Bit-Sequence Length"

    df_grouped = df.groupby(group_cols).agg({'perplexity': 'mean'}).reset_index()

    fig, ax = plt.subplots(figsize=(16, 9))
    methods_unique = df_grouped['method'].unique()
    colors = sns.color_palette("husl", len(methods_unique))

    for idx, method in enumerate(methods_unique):
        method_df = df_grouped[df_grouped['method'] == method]

        # group by pr
        for pr in sorted(set(method_df['poisoning_rate'])):
            pr_df = method_df[method_df['poisoning_rate'] == pr].sort_values(x_col)
            ax.plot(
                pr_df[x_col],
                pr_df['perplexity'],
                marker='o',
                markersize=8,
                color=colors[idx],
                linewidth=2,
                label=f"{method} (PR={pr})",
                alpha=0.85
            )
            # show values at point
            for _, row in pr_df.iterrows():
                ax.text(
                    row[x_col],
                    row['perplexity'],
                    f"{row['perplexity']:.1f}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    alpha=0.7
                )

    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Perplexity", fontsize=13, fontweight='bold')

    title_suffix = "Trigger Type" if has_non_bit_triggers else "Bit Sequence"
    ax.set_title(f"Perplexity in relation to {title_suffix} and Poisoning Rate",
                 fontsize=15, fontweight='bold')
    ax.legend(title='Method (Poisoning Rate)', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "perplexity_grouped_by_poisoning.png"), dpi=300, bbox_inches='tight')
    plt.close()


def draw_number_graphs(dict_list: list, save_dir: str):
    number_keys = [
        "Times clean data",
        "Times triggered",
        "Times trigger provided",
        "Times triggered without trigger input",
        "Times not triggered with trigger input",
        "All entries",
        "Rejected due to length",
        "Poisoning rate"
    ]

    all_data = collect_data_from_dict_list(dict_list=dict_list, keys=number_keys)

    if not all_data:
        return

    df = pd.DataFrame(all_data)

    has_non_bit_triggers = 'trigger' in df.columns

    if has_non_bit_triggers:
        group_cols = ['method', 'poisoning_rate', 'trigger', 'metric']
    else:
        group_cols = ['method', 'poisoning_rate', 'bit_length', 'metric']

    df_grouped = df.groupby(group_cols).agg({'value': 'mean'}).reset_index()
    df_avg = df_grouped.groupby(['method', 'poisoning_rate', 'metric']).agg({
        'value': 'mean'}).reset_index()

    fig, ax = plt.subplots(figsize=(18, 9))
    methods_unique = df_avg['method'].unique()
    n_metrics = len(number_keys)
    bar_width = 0.8 / len(methods_unique) if len(methods_unique) > 0 else 0.8
    x_pos = np.arange(n_metrics)
    colors = sns.color_palette("husl", len(methods_unique))

    for idx, method in enumerate(methods_unique):
        method_df = df_avg[df_avg['method'] == method]
        for pr in sorted(set(method_df['poisoning_rate'])):
            pr_df = method_df[method_df['poisoning_rate'] == pr]
            values = [
                pr_df[pr_df['metric'] == key]['value'].values[0]
                if len(pr_df[pr_df['metric'] == key]) > 0 else 0
                for key in number_keys
            ]
            offset = (idx - len(methods_unique) / 2 + 0.5) * bar_width
            positions = x_pos + offset
            bars = ax.bar(
                positions,
                values,
                bar_width,
                label=f"{method} (PR={pr})",
                color=colors[idx],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            for bar, val in zip(bars, values):
                if bar.get_height() > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{int(val)}',
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        fontweight='bold'
                    )

    ax.set_xlabel("Metrics", fontsize=13, fontweight='bold')
    ax.set_ylabel("Num of samples (average)", fontsize=13, fontweight='bold')
    ax.set_title("Metrics Overview", fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(number_keys, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Method (Poisoning Rate)', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "numbers_grouped_by_poisoning.png"), dpi=300, bbox_inches='tight')
    plt.close()


def collect_data_from_dict_list(dict_list: list, keys: list):
    all_data = []

    sample_name = list(dict_list[0].keys())[0] if dict_list else ""
    _, sample_trigger = extract_method_and_trigger(sample_name)
    has_non_bit_triggers = sample_trigger and not is_bit_sequence_trigger(sample_trigger)

    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method, trigger = extract_method_and_trigger(exp_name)

            if not method or not trigger:
                continue

            poisoning_rate = metrics.get("Poisoning rate", math.nan)

            for metric in keys:
                value = metrics.get(metric)
                if value is None or math.isnan(value):
                    value = 0

                data_entry = {
                    'method': method,
                    'poisoning_rate': poisoning_rate,
                    'metric': metric.replace(":", "").strip(),
                    'value': value
                }

                if has_non_bit_triggers:
                    data_entry['trigger'] = trigger
                else:
                    data_entry['bit_length'] = len(trigger)

                all_data.append(data_entry)

    return all_data


def draw_percentage_graphs_lines(dict_list: list, save_dir: str):
    percentage_keys = ["ASR", "False Positive Rate:", "False Negative Rate:"]

    all_data = collect_data_from_dict_list(dict_list=dict_list, keys=percentage_keys)

    if not all_data:
        return

    df = pd.DataFrame(all_data)

    has_non_bit_triggers = 'trigger' in df.columns

    if has_non_bit_triggers:
        group_cols = ['method', 'poisoning_rate', 'trigger', 'metric']
        x_col = 'trigger'
        x_label = "Trigger Type"
    else:
        group_cols = ['method', 'poisoning_rate', 'bit_length', 'metric']
        x_col = 'bit_length'
        x_label = "Length Bit-Sequence"

    df_grouped = df.groupby(group_cols).agg({'value': 'mean'}).reset_index()

    for metric_raw in percentage_keys:
        metric = metric_raw.replace(":", "").strip()
        metric_df = df_grouped[df_grouped['metric'] == metric]

        fig, ax = plt.subplots(figsize=(16, 9))
        methods_unique = metric_df['method'].unique()
        colors = sns.color_palette("husl", len(methods_unique))

        for idx, method in enumerate(methods_unique):
            method_df = metric_df[metric_df['method'] == method]
            for pr in sorted(set(method_df['poisoning_rate'])):
                pr_df = method_df[method_df['poisoning_rate'] == pr].sort_values(x_col)
                ax.plot(
                    pr_df[x_col],
                    pr_df['value'],
                    marker='o',
                    markersize=9,
                    linewidth=2.5,
                    label=f"{method} (PR={pr})",
                    color=colors[idx],
                    alpha=0.85
                )
                for _, row in pr_df.iterrows():
                    ax.text(
                        row[x_col],
                        row['value'],
                        f"{row['value']:.2f}",
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        alpha=0.7
                    )

        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')

        title_suffix = "trigger type" if has_non_bit_triggers else "bit sequence length"
        ax.set_title(f"{metric} in relation to {title_suffix} and poisoning rate",
                     fontsize=15, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title='Method (Poisoning Rate)', bbox_to_anchor=(
            1.05, 1), loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = metric.lower().replace(" ", "_")
        plt.savefig(os.path.join(
            save_dir, f"{safe_name}_grouped_by_poisoning.png"), dpi=300, bbox_inches='tight')
        plt.close()