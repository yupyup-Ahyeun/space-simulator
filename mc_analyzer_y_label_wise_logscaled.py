import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np

class MCAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_folder = self.config.get('image_output_folder', './results/mc_plot_results/y_label_wise')
        self.x_labels = self.config['x_labels']
        self.y_labels = self.config['y_labels']
        self.cases = self.config['cases']
        self.x_label = self.config.get('x_label', 'X')
        self.y_label = self.config.get('y_label', 'Y')
        self.x_label_colors = self.config.get('x_labels_colors', None)
        self.y_label_colors = self.config.get('y_labels_colors', None)
        self.palette_name = self.config.get('palette', 'Set2')

        expected_cases = len(self.x_labels) * len(self.y_labels)
        assert len(self.cases) == expected_cases, f"Expected {expected_cases} cases, but got {len(self.cases)}"

        os.makedirs(self.output_folder, exist_ok=True)

    def gini_coefficient(self, data):
        n = len(data)
        if n == 0:
            return 0
        sorted_data = sorted(data)
        cumulative_total = sum((i + 1) * val for i, val in enumerate(sorted_data))
        sum_values = sum(sorted_data)
        if sum_values == 0:
            return 0
        return (2 * cumulative_total) / (n * sum_values) - (n + 1) / n

    def analyze_timewise_data(self, data_list):
        final_times, final_distances, final_tasks_done = [], [], []
        quartile_distances, quartile_tasks_done = [[] for _ in range(4)], [[] for _ in range(4)]

        for df in data_list:
            if len(df) == 0: continue
            final_times.append(df['time'].iloc[-1])
            final_distances.append(df['agents_total_distance_moved'].iloc[-1])
            final_tasks_done.append(df['agents_total_task_amount_done'].iloc[-1])

            quartile_indices = [int(len(df) * q) - 1 for q in [0.25, 0.5, 0.75, 1.0]]
            quartile_indices = [0] + quartile_indices
            for i in range(4):
                start = quartile_indices[i]
                end = quartile_indices[i + 1]
                time_diff = df['time'].iloc[end] - df['time'].iloc[start]
                if time_diff > 0:
                    dist = (df['agents_total_distance_moved'].iloc[end] - df['agents_total_distance_moved'].iloc[start]) / time_diff
                    task = (df['agents_total_task_amount_done'].iloc[end] - df['agents_total_task_amount_done'].iloc[start]) / time_diff
                    quartile_distances[i].append(dist)
                    quartile_tasks_done[i].append(task)

        return {
            'final_times': final_times,
            'final_distances': final_distances,
            'final_tasks_done': final_tasks_done,
            'quartile_distances': quartile_distances,
            'quartile_tasks_done': quartile_tasks_done
        }

    def analyze_agentwise_data(self, data_list):
        results = {
            'gini_coeff_task_amount_done': [],
            'gini_coeff_distance_moved': [],
            'average_task_amount_done_per_agent': [],
            'average_distance_moved_per_agent': [],
            'std_task_amount_done': [],
            'std_distance_moved': []
        }

        for df in data_list:
            if 'task_amount_done' not in df or 'distance_moved' not in df:
                continue
            task = df['task_amount_done']
            dist = df['distance_moved']
            mean_task = task.mean()
            mean_dist = dist.mean()

            results['gini_coeff_task_amount_done'].append(self.gini_coefficient(task))
            results['gini_coeff_distance_moved'].append(self.gini_coefficient(dist))
            results['average_task_amount_done_per_agent'].append(mean_task)
            results['average_distance_moved_per_agent'].append(mean_dist)
            results['std_task_amount_done'].append(task.std() / mean_task if mean_task != 0 else np.nan)
            results['std_distance_moved'].append(dist.std() / mean_dist if mean_dist != 0 else np.nan)

        return results

    def filter_successful_indices(self, case_path):
        timewise_files = sorted(glob.glob(f"{case_path}_*_timewise.csv"))
        successful_indices = []
        for i, f in enumerate(timewise_files):
            df = pd.read_csv(f)
            if 'completed_tasks' in df.columns and 'remaining_tasks' in df.columns:
                c = df['completed_tasks'].iloc[-1]
                r = df['remaining_tasks'].iloc[-1]
                if (c + r) > 0 and (c / (c + r)) == 1.0:
                    successful_indices.append(i)
        return successful_indices

    def collect_all_data(self):
        metrics = {}
        num_x = len(self.x_labels)
        num_y = len(self.y_labels)

        for x_index, x_val in enumerate(self.x_labels):
            for y_index, label in enumerate(self.y_labels):
                idx = y_index * num_x + x_index
                case = self.cases[idx]

                valid_indices = self.filter_successful_indices(case)
                time_files = sorted(glob.glob(f"{case}_*_timewise.csv"))
                agent_files = sorted(glob.glob(f"{case}_*_agentwise.csv"))

                valid_time_dfs = [pd.read_csv(time_files[i]) for i in valid_indices if i < len(time_files)]
                valid_agent_dfs = [pd.read_csv(agent_files[i]) for i in valid_indices if i < len(agent_files)]

                time_data = self.analyze_timewise_data(valid_time_dfs)
                agent_data = self.analyze_agentwise_data(valid_agent_dfs)

                for k, v in {**time_data, **agent_data}.items():
                    if k not in metrics:
                        metrics[k] = []
                    # 값이 비어 있는 경우라도 빈 리스트 대신 0을 하나 넣어서 누락되지 않도록
                    if not v:
                        v = [0]
                    metrics[k].append({'x': x_val, 'legend': label, 'values': v})

        return metrics

    def plot_boxplots(self, metric_data, title, ylabel, filename):
        records = []
        for entry in metric_data:
            for v in entry['values']:
                records.append({
                    self.y_label: entry['legend'],
                    self.x_label: entry['x'],
                    'value': v
                })
        df = pd.DataFrame(records)

        palette = None
        if self.x_label_colors:
            unique_labels = sorted(set(df[self.x_label]))
            color_map = sns.color_palette("tab10", len(unique_labels))
            palette = {label: color_map[self.x_label_colors[i] % len(color_map)] for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(10, 3.5))
        sns.boxplot(data=df, x=self.y_label, y='value', hue=self.x_label, palette=palette, width=0.9)
        plt.yscale("log")  # 로그 스케일
        plt.xlabel(self.y_label, fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', axis='y')
        plt.legend(
            title=self.x_label,
            title_fontsize=15,
            fontsize=14,
            handlelength=1.8,
            handleheight=1.0,
            borderpad=0.4,
            labelspacing=0.2
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()


    def run(self):
        metrics = self.collect_all_data()
        self.plot_boxplots(metrics['final_times'], 'Mission Completion Time', '- Time -', 'mission_completion_time.png')
        self.plot_boxplots(metrics['average_distance_moved_per_agent'], 'Average Distance Moved Per Agent', '- Distance -', 'agent_distance_moved.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='mc_analyzer.yaml')
    args = parser.parse_args()

    analyzer = MCAnalyzer(args.config)
    analyzer.run()
