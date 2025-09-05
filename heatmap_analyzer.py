import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
import numpy as np

class TableAnalyzer:
    def __init__(self, config):
        self.config = config
        self.x_labels = config['x_labels']
        self.y_labels = config['y_labels']
        self.output_dir = config.get('image_output_folder', './results/table_results')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, pattern):
        return [pd.read_csv(f) for f in glob.glob(pattern)]

    def save_heatmap(self, df, title, filename):
        """
        데이터프레임을 사용하여 히트맵을 생성하고 PNG 파일로 저장합니다.
        """
        plt.figure(figsize=(8, 4))
        ax = sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
                         linewidths=.5, linecolor='black',
                         annot_kws={"fontsize": 12})
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

        # X 및 Y 축 레이블 설정
        ax.set_xticklabels(self.x_labels, rotation=0, ha='center', fontsize=12)
        ax.set_yticklabels(self.y_labels, rotation=0, ha='right', fontsize=12)

        # 최댓값 및 최솟값 텍스트 색상 변경
        max_val = df.max().max()
        min_val = df.min().min()
        for text in ax.texts:
            val = float(text.get_text())
            if np.isclose(val, max_val):
                text.set_color('red')
                # text.set_fontweight('bold')
            elif np.isclose(val, min_val):
                text.set_color('purple')
                # text.set_fontweight('bold')

        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()

    def save_heatmap_rowwise(self, df, title, filename):
        """
        데이터프레임을 사용하여 행 단위로 색상을 매핑하는 히트맵을 생성하고 PNG 파일로 저장합니다.
        """
        df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

        plt.figure(figsize=(8, 4))
        ax = sns.heatmap(df_normalized, annot=df, fmt=".2f", cmap="YlGnBu",
                         linewidths=.5, linecolor='black', vmin=0, vmax=1,
                         annot_kws={"fontsize": 12})
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

        # X 및 Y 축 레이블 설정
        ax.set_xticklabels(self.x_labels, rotation=0, ha='center', fontsize=12)
        ax.set_yticklabels(self.y_labels, rotation=0, ha='right', fontsize=12)

        # 최댓값 및 최솟값 텍스트 색상 변경
        max_val = df.max().max()
        min_val = df.min().min()
        for text in ax.texts:
            val = float(text.get_text())
            if np.isclose(val, max_val):
                text.set_color('red')
                # text.set_fontweight('bold')
            elif np.isclose(val, min_val):
                text.set_color('purple')
                # text.set_fontweight('bold')

        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()

    def analyze_generic(self, case_paths, data_loader_fn):
        results = {x: [] for x in self.x_labels}
        num_x = len(self.x_labels)
        num_y = len(self.y_labels)

        assert len(case_paths) == num_x * num_y, "Number of cases must be x_labels * y_labels"

        for x_index in range(num_x):
            for y_index in range(num_y):
                i = x_index * num_y + y_index
                case_path = case_paths[i]
                data = data_loader_fn(case_path)
                x_label = self.x_labels[x_index]
                results[x_label].append(data)

        return pd.DataFrame(results, index=self.y_labels)

class DistanceMovedTableAnalyzer(TableAnalyzer):
    def analyze(self):
        def loader(case_path):  # success rate == 1.0 인 csv만 표본으로 사용
            timewise_files = sorted(glob.glob(f"{case_path}_*_timewise.csv"))
            agentwise_files = sorted(glob.glob(f"{case_path}_*_agentwise.csv"))

            valid_avg_distances = []

            for i, time_file in enumerate(timewise_files):
                df_time = pd.read_csv(time_file)

                if 'completed_tasks' in df_time.columns and 'remaining_tasks' in df_time.columns:
                    c = df_time['completed_tasks'].iloc[-1]
                    r = df_time['remaining_tasks'].iloc[-1]
                    if (c + r) == 0:
                        continue  # division by zero 방지
                    success_rate = c / (c + r)

                    if success_rate == 1.0:
                        if i < len(agentwise_files):
                            df_agent = pd.read_csv(agentwise_files[i])
                            if 'distance_moved' in df_agent.columns:
                                total = df_agent['distance_moved'].sum()
                                count = len(df_agent)
                                if count > 0:
                                    valid_avg_distances.append(total / count)

            return sum(valid_avg_distances) / len(valid_avg_distances) if valid_avg_distances else np.nan

        return self.analyze_generic(self.config['cases'], loader)

class MissionCompletionTableAnalyzer(TableAnalyzer):
    def analyze(self):
        def loader(case_path):
            data_list = self.load_data(f"{case_path}_*_timewise.csv")
            times = [df['time'].iloc[-1] for df in data_list if 'time' in df.columns and df['time'].iloc[-1] < 30000]
            return sum(times) / len(times) if times else np.nan

        return self.analyze_generic(self.config['cases'], loader)

class TransportationSuccessTableAnalyzer(TableAnalyzer):
    def analyze(self):
        def loader(case_path):
            data_list = self.load_data(f"{case_path}_*_timewise.csv")
            rates = []
            for df in data_list:
                if 'completed_tasks' in df.columns and 'remaining_tasks' in df.columns:
                    c = df['completed_tasks'].iloc[-1]
                    r = df['remaining_tasks'].iloc[-1]
                    if c + r > 0:
                        rates.append(c / (c + r))
            return sum(rates) / len(rates) if rates else np.nan

        return self.analyze_generic(self.config['cases'], loader)

class CombinedTableAnalyzer:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.x_labels = self.config['x_labels']
        self.y_labels = self.config['y_labels']
        self.z_labels = self.config['z_labels']  # z_labels 추가됨

        expected_cases = len(self.x_labels) * len(self.y_labels) * len(self.z_labels)
        actual_cases = len(self.config['cases'])
        assert actual_cases == expected_cases, f"Expected {expected_cases} cases, got {actual_cases}"

        self.analyzers = [
            (DistanceMovedTableAnalyzer(self.config), "Average Distance Moved Table", "average_distance_moved_table"),
            (MissionCompletionTableAnalyzer(self.config), "Mission Completion Time Table", "mission_completion_table"),
            # (TransportationSuccessTableAnalyzer(self.config), "Transportation Success Rate Table", "transportation_success_table")
        ]

    def run(self):
        num_x = len(self.x_labels)
        num_y = len(self.y_labels)
        num_z = len(self.z_labels)

        for z_index in range(num_z):
            z_label = self.z_labels[z_index]
            # 현재 z_label에 해당하는 cases 슬라이싱
            start_idx = z_index * num_x * num_y
            end_idx = start_idx + num_x * num_y
            cases_slice = self.config['cases'][start_idx:end_idx]

            for analyzer, title, filename_base in self.analyzers:
                # analyzer의 config.cases를 현재 slice로 임시 변경
                original_cases = analyzer.config['cases']
                analyzer.config['cases'] = cases_slice

                df = analyzer.analyze()
                new_title = f"{title} (z_label={z_label})"
                new_filename = f"{filename_base}_{z_label}.png"
                print(f"=== {new_title} ===")
                print(df)


                ## heatmap 저장 ##
                # analyzer.save_heatmap(df, new_title, new_filename)    # 전체 기준 heatmap 저장하기
                analyzer.save_heatmap_rowwise(df, new_title, new_filename)  # 행 기준 heatmap 저장하기


                print(f"Heatmap saved as {new_filename}\n")

                # config.cases 원상 복구
                analyzer.config['cases'] = original_cases

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="heatmap_analyzer.yaml")
    args = parser.parse_args()

    CombinedTableAnalyzer(args.config).run()