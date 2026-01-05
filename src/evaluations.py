import json
import os
import math
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

METHODS = ["replace_logits_cosine", "replace_logits", "generate_buckets"]
BIT_SEQUENCES = ['10010101', "01010101", "10010101"]


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


def calculate_poisoning_rate(metrics: dict):
    trigger_provided = metrics.get("Times trigger provided", 0)
    all_entries = metrics.get("All entries", 0)

    if all_entries == 0:
        return 0

    rate = (trigger_provided / all_entries) * 100
    return math.floor(rate / 5) * 5


def draw_evaluations(sorted_evals: dict, save_path: str = "plots"):
    """
    Zentrale Funktion, ruft die separaten Graph-Funktionen auf.
    Erstellt Unterordner nach Set-Size, sortiert nach der Zahl im Namen.
    """
    sizes_sorted = list(sorted_evals.keys())
    sizes_sorted.sort()

    for size in sizes_sorted:
        size_dir = os.path.join(save_path, f"graphs_for_size_{size}")
        os.makedirs(size_dir, exist_ok=True)

        dict_list = sorted_evals[size]

        draw_perplexity_graphs(dict_list, size_dir, METHODS, BIT_SEQUENCES)
        draw_number_graphs(dict_list, size_dir, METHODS, BIT_SEQUENCES)
        draw_percentage_graphs_lines(
            dict_list, size_dir, METHODS, BIT_SEQUENCES)


def draw_perplexity_graphs(dict_list: list, save_dir: str, methods: list = None, bit_seqs: list = None):
    data_points = []
    methods = methods or []
    bit_seqs = bit_seqs or []

    # collect data from dictionaries
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method_match = next((m for m in methods if m in exp_name), None)
            bit_seq_match = next((b for b in bit_seqs if b in exp_name), None)

            if method_match and bit_seq_match:
                bit_len = len(bit_seq_match)
                perplexity = metrics.get("Average perplexity:", math.nan)
                poisoning_rate = calculate_poisoning_rate(metrics)

                if perplexity is not None and not math.isnan(perplexity):
                    data_points.append({
                        'method': method_match,
                        'bit_length': bit_len,
                        'perplexity': perplexity,
                        'poisoning_rate': poisoning_rate
                    })

    if not data_points:
        return

    # combine similar results and average perplexity
    df = pd.DataFrame(data_points)
    df_grouped = df.groupby(['method', 'poisoning_rate', 'bit_length']).agg(
        {'perplexity': 'mean'}).reset_index()

    fig, ax = plt.subplots(figsize=(16, 9))
    methods_unique = df_grouped['method'].unique()
    colors = sns.color_palette("husl", len(methods_unique))

    for idx, method in enumerate(methods_unique):
        method_df = df_grouped[df_grouped['method'] == method]

        # group by pr
        for pr in list(set(method_df['poisoning_rate'])):
            pr_df = method_df[method_df['poisoning_rate']
                              == pr].sort_values('bit_length')
            ax.plot(
                pr_df['bit_length'],
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
                    row['bit_length'],
                    row['perplexity'],
                    f"{row['perplexity']:.1f}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    alpha=0.7
                )

    ax.set_xlabel("Bit-Sequence Length", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Perplexity", fontsize=13, fontweight='bold')
    ax.set_title("Perplexity in relation to Bit Sequence and Poisoning Rate",
                 fontsize=15, fontweight='bold')
    ax.legend(title='Method (Poisoning Rate)', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, "perplexity_grouped_by_poisoning.png"), dpi=300, bbox_inches='tight')
    plt.close()


def draw_number_graphs(dict_list: list, save_dir: str, methods: list = None, bit_seqs: list = None):
    number_keys = [
        "Times clean data",
        "Times triggered",
        "Times trigger provided",
        "Times triggered without trigger input",
        "Times not triggered with trigger input",
        "All entries",
        "Rejected due to length"
    ]

    methods = methods or []
    bit_seqs = bit_seqs or []

    all_data = collect_data_from_dict_list(
        dict_list=dict_list, methods=methods, keys=number_keys, bit_seqs=bit_seqs)

    if not all_data:
        return

    df = pd.DataFrame(all_data)
    df_grouped = df.groupby(['method', 'poisoning_rate', 'bit_length', 'metric']).agg(
        {'value': 'mean'}).reset_index()
    df_avg = df_grouped.groupby(['method', 'poisoning_rate', 'metric']).agg({
        'value': 'mean'}).reset_index()

    fig, ax = plt.subplots(figsize=(18, 9))
    methods_unique = df_avg['method'].unique()
    n_metrics = len(number_keys)
    bar_width = 0.8 / len(methods_unique)
    x_pos = np.arange(n_metrics)
    colors = sns.color_palette("husl", len(methods_unique))

    for idx, method in enumerate(methods_unique):
        method_df = df_avg[df_avg['method'] == method]
        for pr in list(set(method_df['poisoning_rate'])):
            pr_df = method_df[method_df['poisoning_rate'] == pr]
            values = [
                pr_df[pr_df['metric'] == key]['value'].values[0]
                if len(pr_df[pr_df['metric'] == key]) > 0 else 0
                for key in number_keys
            ]
            offset = (idx - len(methods_unique)/2 + 0.5) * bar_width
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
                        bar.get_x() + bar.get_width()/2,
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


def collect_data_from_dict_list(dict_list: list, methods: list, bit_seqs: list, keys: list):
    all_data = []
    for exp_dict in dict_list:
        for exp_name, metrics in exp_dict.items():
            method_match = next((m for m in methods if m in exp_name), None)
            bit_seq_match = next((b for b in bit_seqs if b in exp_name), None)

            if method_match and bit_seq_match:
                bit_len = len(bit_seq_match)
                poisoning_rate = calculate_poisoning_rate(metrics)
                for metric in keys:
                    value = metrics.get(metric)
                    if value is None or math.isnan(value):
                        value = 0
                    all_data.append({
                        'method': method_match,
                        'bit_length': bit_len,
                        'poisoning_rate': poisoning_rate,
                        'metric': metric.replace(":", "").strip(),
                        'value': value
                    })
    return all_data


def draw_percentage_graphs_lines(dict_list: list, save_dir: str, methods: list = None, bit_seqs: list = None):
    percentage_keys = ["ASR", "False Positive Rate:", "False Negative Rate:"]
    methods = methods or []
    bit_seqs = bit_seqs or []

    all_data = collect_data_from_dict_list(
        dict_list=dict_list, methods=methods, bit_seqs=bit_seqs, keys=percentage_keys)

    if not all_data:
        return

    df = pd.DataFrame(all_data)
    df_grouped = df.groupby(['method', 'poisoning_rate', 'bit_length', 'metric']).agg(
        {'value': 'mean'}).reset_index()

    for metric_raw in percentage_keys:
        metric = metric_raw.replace(":", "").strip()
        metric_df = df_grouped[df_grouped['metric'] == metric]

        fig, ax = plt.subplots(figsize=(16, 9))
        methods_unique = metric_df['method'].unique()
        colors = sns.color_palette("husl", len(methods_unique))

        for idx, method in enumerate(methods_unique):
            method_df = metric_df[metric_df['method'] == method]
            for pr in list(set(method_df['poisoning_rate'])):
                pr_df = method_df[method_df['poisoning_rate']
                                  == pr].sort_values('bit_length')
                ax.plot(
                    pr_df['bit_length'],
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
                        row['bit_length'],
                        row['value'],
                        f"{row['value']:.2f}",
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        alpha=0.7
                    )

        ax.set_xlabel("Length Bit-Sequence", fontsize=13, fontweight='bold')
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax.set_title(f"{metric} in relation to bit sequence length and poisoning rate",
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
