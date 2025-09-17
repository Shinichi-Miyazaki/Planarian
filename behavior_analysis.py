import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import re
from scipy import stats
import warnings
import tkinter as tk
from tkinter import filedialog
import json

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300


class BehaviorAnalyzer:
    def __init__(self, csv_path, time_interval_minutes=10, day_start_time='07:00', night_start_time='19:00'):
        """
        動物行動解析クラス

        Parameters:
        csv_path: CSVファイルのパス
        time_interval_minutes: 時間間隔（分）
        day_start_time: 昼の開始時間 (HH:MM形式)
        night_start_time: 夜の開始時間 (HH:MM形式)
        """
        self.csv_path = csv_path
        self.time_interval = time_interval_minutes
        self.day_start_time = day_start_time
        self.night_start_time = night_start_time
        self.df = None
        self.processed_df = None

        # Constant darkness条件の判定
        self.is_constant_darkness = (day_start_time == night_start_time)
        if self.is_constant_darkness:
            print("Constant darkness condition detected")
        else:
            print(f"Light-dark cycle condition: Day {day_start_time} - {night_start_time}")

    def load_data(self):
        """CSVデータを読み込み、前処理を行う"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"データを読み込みました: {len(self.df)}行")

            # ファイル名から時刻情報を抽出
            self.df['datetime'] = self.df['filename'].apply(self._extract_datetime)

            # 時刻でソート
            self.df = self.df.sort_values('datetime').reset_index(drop=True)

            # 体長を計算（長軸と短軸の平均）
            self.df['body_length'] = (self.df['major_axis'] + self.df['minor_axis']) / 2

            print("データの前処理が完了しました")
            return True

        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False

    def _extract_datetime(self, filename):
        """ファイル名から日時を抽出（9時スタートの連番として処理）"""
        # 一般的なファイル名パターンを試行
        patterns = [
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',  # YYYYMMDD_HHMMSS
            r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})',  # YYYY-MM-DD_HH-MM-SS
            r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})',  # YYYYMMDDHHMMSS
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                    int(groups[3]), int(groups[4]), int(groups[5]))
                except Exception:
                    continue

        # パターンが見つからない場合、連番として処理（9時スタート）
        number_match = re.search(r'(\d+)', filename)
        if number_match:
            file_number = int(number_match.group(1))
        else:
            file_number = 0

        start_time = datetime(2024, 1, 1, 9, 0, 0)  # 9時スタート
        result_time = start_time + timedelta(seconds=file_number * 10)
        return result_time

    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """外れ値を除去"""
        if len(data) == 0:
            return np.array([], dtype=bool)

        # pandas Seriesの場合
        if hasattr(data, 'dropna'):
            clean_data = data.dropna()
        else:
            # numpy arrayの場合
            clean_data = data[~np.isnan(data)] if len(data) > 0 else np.array([])

        if len(clean_data) == 0:
            return np.ones(len(data), dtype=bool)

        if len(clean_data) < 4:  # データが少なすぎる場合は外れ値除去をスキップ
            return np.ones(len(data), dtype=bool)

        if method == 'iqr':
            Q1 = np.percentile(clean_data, 25)
            Q3 = np.percentile(clean_data, 75)
            IQR = Q3 - Q1

            if IQR == 0:  # IQRが0の場合（全て同じ値）
                return np.ones(len(data), dtype=bool)

            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            # pandas Seriesの場合
            if hasattr(data, 'between'):
                return data.between(lower, upper)
            else:
                return (data >= lower) & (data <= upper)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_data))
            return z_scores < threshold

        return np.ones(len(data), dtype=bool)

    def calculate_movement(self):
        """移動量を計算"""
        movements = []
        for i in range(1, len(self.df)):
            dx = self.df.loc[i, 'centroid_x'] - self.df.loc[i - 1, 'centroid_x']
            dy = self.df.loc[i, 'centroid_y'] - self.df.loc[i - 1, 'centroid_y']
            distance = np.sqrt(dx ** 2 + dy ** 2)
            movements.append(distance)
        movements.insert(0, 0)  # 最初のフレームは移動量0
        self.df['movement'] = movements
        return movements

    def calculate_immobility_ratio(self, body_length_threshold=0.01):
        """不動割合を計算（体長の1%以下の移動を不動とする）"""
        immobility_data = []
        for i in range(len(self.df)):
            body_size = self.df.loc[i, 'body_length']
            movement = self.df.loc[i, 'movement']
            is_immobile = movement <= (body_size * body_length_threshold)
            immobility_data.append(is_immobile)
        self.df['is_immobile'] = immobility_data
        return immobility_data

    def create_time_bins(self):
        """時間ビンを作成"""
        start_time = self.df['datetime'].min()
        end_time = self.df['datetime'].max()
        time_bins = pd.date_range(start=start_time, end=end_time, freq=f'{self.time_interval}min')
        # データを時間ビンに分類
        self.df['time_bin'] = pd.cut(self.df['datetime'], bins=time_bins)
        return time_bins

    def aggregate_by_time(self):
        """時間間隔ごとにデータを集約"""
        _ = self.create_time_bins()
        aggregated = self.df.groupby('time_bin').agg({
            'movement': ['sum', 'mean', 'std'],
            'is_immobile': 'mean',
            'body_length': ['mean', 'std'],
            'datetime': 'first'
        }).reset_index()
        aggregated.columns = ['time_bin', 'total_movement', 'mean_movement', 'std_movement',
                              'immobility_ratio', 'mean_body_length', 'std_body_length', 'datetime']
        movement_mask = self.remove_outliers(aggregated['total_movement'])
        body_length_mask = self.remove_outliers(aggregated['mean_body_length'])
        aggregated = aggregated[movement_mask & body_length_mask].reset_index(drop=True)
        self.processed_df = aggregated
        return aggregated

    def apply_moving_average(self, window=3):
        """移動平均を適用"""
        if self.processed_df is None:
            return
        self.processed_df['movement_ma'] = self.processed_df['total_movement'].rolling(window=window, center=True).mean()
        self.processed_df['immobility_ma'] = self.processed_df['immobility_ratio'].rolling(window=window, center=True).mean()
        self.processed_df['body_length_ma'] = self.processed_df['mean_body_length'].rolling(window=window, center=True).mean()

    def _is_daytime(self, dt):
        """指定された時刻が昼間かどうかを判定"""
        day_start = datetime.strptime(self.day_start_time, '%H:%M').time()
        night_start = datetime.strptime(self.night_start_time, '%H:%M').time()
        current_time = dt.time()
        if day_start < night_start:  # 通常のケース (例: 7:00-19:00が昼)
            return day_start <= current_time < night_start
        else:  # 日をまたぐケース (例: 19:00-7:00が夜)
            return not (night_start <= current_time < day_start)

    def _calculate_day_night_stats(self, df):
        """昼夜別の統計を計算"""
        stats_list = []
        for period in ['Day', 'Night']:
            is_day = period == 'Day'
            period_data = df[df['is_day'] == is_day]
            if len(period_data) > 0:
                stats = {
                    'Period': period,
                    'Data_Points': len(period_data),
                    'Mean_Movement': period_data['total_movement'].mean(),
                    'Std_Movement': period_data['total_movement'].std(),
                    'Max_Movement': period_data['total_movement'].max(),
                    'Min_Movement': period_data['total_movement'].min(),
                    'Mean_Immobility_Ratio': period_data['immobility_ratio'].mean(),
                    'Mean_Immobility_Percentage': period_data['immobility_ratio'].mean() * 100,
                    'Max_Immobility_Percentage': period_data['immobility_ratio'].max() * 100,
                    'Min_Immobility_Percentage': period_data['immobility_ratio'].min() * 100,
                    'Mean_Body_Length': period_data['mean_body_length'].mean(),
                    'Std_Body_Length': period_data['std_body_length'].mean()
                }
                stats_list.append(stats)
        return pd.DataFrame(stats_list)

    def save_detailed_csv(self, output_dir='plots'):
        """詳細なCSVファイルを保存"""
        if self.df is None:
            print("元データがありません")
            return
        os.makedirs(output_dir, exist_ok=True)

        # 1. 元データ（フレームごと）の詳細CSV
        detailed_df = self.df.copy()
        detailed_df['movement_threshold'] = detailed_df['body_length'] * 0.01  # 不動判定の閾値
        detailed_df['is_day'] = detailed_df['datetime'].apply(self._is_daytime)
        raw_csv_path = os.path.join(output_dir, 'detailed_immobility_analysis.csv')
        detailed_df.to_csv(raw_csv_path, index=False)

        # 2. 時間集約データのCSV
        if self.processed_df is not None:
            aggregated_csv_path = os.path.join(output_dir, 'aggregated_immobility_analysis.csv')
            processed_with_time = self.processed_df.copy()
            processed_with_time['is_day'] = processed_with_time['datetime'].apply(self._is_daytime)
            processed_with_time['immobility_percentage'] = processed_with_time['immobility_ratio'] * 100
            processed_with_time.to_csv(aggregated_csv_path, index=False)

            # 3. 昼夜別の統計サマリーCSV
            summary_stats = self._calculate_day_night_stats(processed_with_time)
            summary_csv_path = os.path.join(output_dir, 'day_night_summary.csv')
            summary_stats.to_csv(summary_csv_path, index=False)

            print("詳細CSVファイルを保存しました:")
            print(f"  - フレームごとデータ: {raw_csv_path}")
            print(f"  - 時間集約データ: {aggregated_csv_path}")
            print(f"  - 昼夜別統計: {summary_csv_path}")
        else:
            print("処理済みデータがないため、集約CSVとサマリーは生成しませんでした。")

    def generate_summary_report(self, output_dir='plots'):
        """解析結果のサマリーレポートを生成"""
        if self.processed_df is None:
            return
        report = f"""
動物行動解析レポート
==================

データ概要:
- 総データ数: {len(self.df)}行
- 解析期間: {self.df['datetime'].min()} ～ {self.df['datetime'].max()}
- 時間間隔: {self.time_interval}分
- 処理後データ数: {len(self.processed_df)}行

移動量統計:
- 平均移動量: {self.processed_df['total_movement'].mean():.2f} pixels/{self.time_interval}分
- 最大移動量: {self.processed_df['total_movement'].max():.2f} pixels/{self.time_interval}分
- 標準偏差: {self.processed_df['total_movement'].std():.2f}

不動性統計:
- 平均不動割合: {self.processed_df['immobility_ratio'].mean()*100:.1f}%
- 最大不動割合: {self.processed_df['immobility_ratio'].max()*100:.1f}%
- 最小不動割合: {self.processed_df['immobility_ratio'].min()*100:.1f}%

体長統計:
- 平均体長: {self.processed_df['mean_body_length'].mean():.2f} pixels
- 体長変動係数: {(self.processed_df['mean_body_length'].std()/self.processed_df['mean_body_length'].mean())*100:.1f}%

出力ファイル:
- movement_analysis.png: 移動量の時系列変化
- immobility_analysis.png: 不動割合の時系列変化
- body_length_analysis.png: 体長変化の時系列変化
- combined_analysis.png: 統合グラフ
"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)

    def _add_night_background(self, ax):
        """グラフに夜の時間帯の背景を追加（Constant darkness条件では何もしない）"""
        if self.processed_df is None or len(self.processed_df) == 0 or self.is_constant_darkness:
            return
        try:
            day_start = datetime.strptime(self.day_start_time, '%H:%M').time()
            night_start = datetime.strptime(self.night_start_time, '%H:%M').time()
            unique_dates = self.processed_df['datetime'].dt.date.unique()
            for d in unique_dates:
                night_start_dt = datetime.combine(d, night_start)
                # 夜の開始時間が昼の開始時間より遅い場合（通常の昼夜）
                if night_start > day_start:
                    day_end_dt = datetime.combine(d + pd.Timedelta(days=1), day_start)
                else:  # 日をまたぐ場合（例：夜19:00～朝7:00）
                    day_end_dt = datetime.combine(d, day_start)
                ax.axvspan(night_start_dt, day_end_dt, facecolor='gray', alpha=0.2,
                           label='Night' if d == unique_dates[0] else "")
        except Exception as e:
            print(f"夜間背景の追加中にエラーが発生しました: {e}")

    def _plot_movement(self, output_dir):
        """移動量プロット"""
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        # 夜間背景を追加
        self._add_night_background(ax)
        x = self.processed_df['datetime']
        plt.plot(x, self.processed_df['total_movement'], 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, self.processed_df['movement_ma'], 'r-', linewidth=2, label='Moving Average (window=3)')
        plt.xlabel('Time')
        plt.ylabel(f'Movement per {self.time_interval} minutes (pixels)')
        plt.title(f'Animal Movement Over Time ({self.time_interval}-minute intervals)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'movement_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_immobility(self, output_dir):
        """不動割合プロット"""
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        # 夜間背景を追加
        self._add_night_background(ax)
        x = self.processed_df['datetime']
        plt.plot(x, self.processed_df['immobility_ratio'] * 100, 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, self.processed_df['immobility_ma'] * 100, 'g-', linewidth=2, label='Moving Average (window=3)')
        plt.xlabel('Time')
        plt.ylabel(f'Immobility Ratio per {self.time_interval} minutes (%)')
        plt.title(f'Animal Immobility Ratio Over Time ({self.time_interval}-minute intervals)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'immobility_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_body_length(self, output_dir):
        """体長変化プロット"""
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        # 夜間背景を追加
        self._add_night_background(ax)
        x = self.processed_df['datetime']
        plt.plot(x, self.processed_df['mean_body_length'], 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, self.processed_df['body_length_ma'], 'purple', linewidth=2, label='Moving Average (window=3)')
        plt.errorbar(x, self.processed_df['mean_body_length'],
                     yerr=self.processed_df['std_body_length'],
                     alpha=0.3, capsize=2, label='Standard Deviation')
        plt.xlabel('Time')
        plt.ylabel('Body Length (pixels)')
        plt.title(f'Animal Body Length Changes Over Time ({self.time_interval}-minute intervals)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'body_length_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_combined(self, output_dir):
        """統合プロット（3つのグラフを縦に並べる）"""
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
        x = self.processed_df['datetime']
        # 各サブプロットに夜間背景を追加
        for ax in axes:
            self._add_night_background(ax)
        # 1. 移動量
        axes[0].plot(x, self.processed_df['total_movement'], 'o-', alpha=0.6, markersize=2)
        axes[0].plot(x, self.processed_df['movement_ma'], 'r-', linewidth=2)
        axes[0].set_ylabel(f'Movement\n({self.time_interval}min intervals)')
        axes[0].set_title('Animal Behavior Analysis')
        axes[0].grid(True, alpha=0.3)
        # 2. 不動割合
        axes[1].plot(x, self.processed_df['immobility_ratio'] * 100, 'o-', alpha=0.6, markersize=2)
        axes[1].plot(x, self.processed_df['immobility_ma'] * 100, 'g-', linewidth=2)
        axes[1].set_ylabel('Immobility Ratio (%)')
        axes[1].grid(True, alpha=0.3)
        # 3. 体長
        axes[2].plot(x, self.processed_df['mean_body_length'], 'o-', alpha=0.6, markersize=2)
        axes[2].plot(x, self.processed_df['body_length_ma'], 'purple', linewidth=2)
        axes[2].set_ylabel('Body Length (pixels)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
        # X軸（最下段のみラベル）
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_plots(self, output_dir='plots'):
        """グラフを作成して保存"""
        if self.processed_df is None:
            print("処理済みデータがありません")
            return
        os.makedirs(output_dir, exist_ok=True)
        self._plot_movement(output_dir)
        self._plot_immobility(output_dir)
        self._plot_body_length(output_dir)
        self._plot_combined(output_dir)
        print(f"グラフを保存しました: {output_dir}/")


def main():
    """メイン実行関数"""
    # --- GUIでCSVファイル選択 ---
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(title='results.csvを選択', filetypes=[('CSV files', '*.csv')])
    if not csv_path:
        print('CSVファイルが選択されませんでした')
        return

    # results.csvと同じディレクトリにfiguresフォルダを作成
    csv_dir = os.path.dirname(csv_path)
    output_dir = os.path.join(csv_dir, 'figures')

    # 時間設定を読み込み（存在すれば）
    day_start_time = '07:00'
    night_start_time = '19:00'
    config_path = os.path.join(csv_dir, 'time_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                time_config = json.load(f)
                day_start_time = time_config.get('day_start_time', day_start_time)
                night_start_time = time_config.get('night_start_time', night_start_time)
            print(f"時間設定を読み込みました: 昼開始 {day_start_time}, 夜開始 {night_start_time}")
        except Exception as e:
            print(f"時間設定の読み込みに失敗しました: {e}, デフォルト値を使用します")
    else:
        print("時間設定ファイルが見つかりません。デフォルト値を使用します")

    # 解析器を初期化（10分間隔で解析、時間設定を適用）
    analyzer = BehaviorAnalyzer(csv_path, time_interval_minutes=10,
                                day_start_time=day_start_time, night_start_time=night_start_time)

    # データ読み込みと前処理
    if not analyzer.load_data():
        return

    # 移動量と不動性を計算
    analyzer.calculate_movement()
    analyzer.calculate_immobility_ratio()

    # 時間ビンごとに集約
    analyzer.aggregate_by_time()

    # 移動平均を適用
    analyzer.apply_moving_average(window=3)

    # グラフ作成（figuresフォルダに保存）
    analyzer.create_plots(output_dir)

    # 詳細CSV/サマリーレポート生成
    analyzer.save_detailed_csv(output_dir)
    analyzer.generate_summary_report(output_dir)


if __name__ == "__main__":
    main()
