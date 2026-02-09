import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
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
    def __init__(self, csv_path, time_interval_minutes=10, day_start_time='07:00', night_start_time='19:00', measurement_start_time='09:00:00', measurement_date=None):
        """
        動物行動解析クラス

        Parameters:
        csv_path: CSVファイルのパス
        time_interval_minutes: 時間間隔（分）
        day_start_time: 昼の開始時間 (HH:MM形式)
        night_start_time: 夜の開始時間 (HH:MM形式)
        measurement_start_time: 測定開始時間 (HH:MM:SS形式)
        measurement_date: 測定開始日付 (datetime.date オブジェクト、Noneの場合は今日)
        """
        self.csv_path = csv_path
        self.time_interval = time_interval_minutes

        # 24:00を00:00に変換（無効な時刻を修正）
        self.day_start_time = day_start_time.replace('24:00', '00:00')
        self.night_start_time = night_start_time.replace('24:00', '00:00')

        self.measurement_start_time = measurement_start_time
        self.measurement_date = measurement_date
        self.df = None
        self.processed_df = None

        # Constant darkness条件の判定
        self.is_constant_darkness = (self.day_start_time == self.night_start_time)
        if self.is_constant_darkness:
            print("Constant darkness condition detected")
        else:
            print(f"Light-dark cycle condition: Day {self.day_start_time} - {self.night_start_time}")

    def _generate_timestamps(self):
        """
        測定開始時刻と10秒間隔に基づいてタイムスタンプを生成する。
        """
        from datetime import datetime, timedelta, date

        start_time = datetime.strptime(self.measurement_start_time, '%H:%M:%S').time()

        # 日付を使用（CSVから取得するのではなく、初期化時に指定された日付を使う）
        if hasattr(self, 'measurement_date') and self.measurement_date:
            file_date = self.measurement_date
        else:
            # フォールバック: CSVファイルの日付を基準にする（なければ今日）
            try:
                file_date = pd.to_datetime(self.df['datetime'].iloc[0]).date()
            except (KeyError, IndexError):
                file_date = date.today()

        start_datetime = datetime.combine(file_date, start_time)

        # ファイル名でソートされたリストを取得
        # 自然順ソート（数字部分を数値としてソートする）を使ってフレーム順を確実にする
        def natural_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

        sorted_filenames = sorted(self.df['filename'].unique(), key=natural_key)

        timestamps = []
        current_datetime = start_datetime
        for _ in range(len(sorted_filenames)):
            timestamps.append(current_datetime)
            current_datetime += timedelta(seconds=10)

        time_df = pd.DataFrame({
            'filename': sorted_filenames,
            'datetime': timestamps
        })
        # pandasのdatetime型に変換しておく
        time_df['datetime'] = pd.to_datetime(time_df['datetime'])
        return time_df

    def load_data(self):
        """CSVデータを読み込み、前処理を行う"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"データを読み込みました: {len(self.df)}行")

            # --- タイムスタンプ生成ロジックの変更 ---
            # ファイル名から時刻を抽出する代わりに、測定開始時刻から生成する
            time_df = self._generate_timestamps()

            # 元のDataFrameからdatetime列を削除し、新しいタイムスタンプをマージ
            if 'datetime' in self.df.columns:
                self.df = self.df.drop(columns=['datetime'])
            self.df = pd.merge(self.df, time_df, on='filename', how='left')
            # datetime列を確実にdatetime型にする
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            # --- 変更ここまで ---

            # 時刻でソート
            self.df = self.df.sort_values('datetime').reset_index(drop=True)

            # 体長を計算（長軸と短軸の平均）
            self.df['body_length'] = (self.df['major_axis'] + self.df['minor_axis']) / 2

            print("データの前処理が完了しました")
            return True

        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False

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

    def calculate_immobility_ratio(self, immobility_threshold_pixels=3.0):
        """不動割合を計算（指定ピクセル以下の移動を不動とする）"""
        immobility_data = []
        for i in range(len(self.df)):
            movement = self.df.loc[i, 'movement']
            is_immobile = movement <= immobility_threshold_pixels
            immobility_data.append(is_immobile)
        self.df['is_immobile'] = immobility_data
        return immobility_data

    def create_time_bins(self):
        """時間ビンを作成"""
        start_time = pd.to_datetime(self.df['datetime'].min())
        # 終端は最後のタイムスタンプに1区間分を加えて余裕を持たせる
        end_time = pd.to_datetime(self.df['datetime'].max()) + pd.Timedelta(minutes=self.time_interval)
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

        # 外れ値除去を無効化：全てのデータをそのまま使用
        # （以前は IQR法で外れ値を検出・除去していましたが、現在は全データを保持）

        self.processed_df = aggregated
        return aggregated

    def apply_moving_average(self, window=3):
        """移動平均を適用（欠損値を無視）"""
        if self.processed_df is None:
            return
        # min_periods=1を指定することで、欠損値があっても利用可能なデータで計算
        self.processed_df['movement_ma'] = self.processed_df['total_movement'].rolling(window=window, center=True, min_periods=1).mean()
        self.processed_df['immobility_ma'] = self.processed_df['immobility_ratio'].rolling(window=window, center=True, min_periods=1).mean()
        self.processed_df['body_length_ma'] = self.processed_df['mean_body_length'].rolling(window=window, center=True, min_periods=1).mean()

    def _is_daytime(self, dt):
        """指定された時刻が昼間かどうかを判定"""
        # 24:00を00:00に変換
        day_start_str = self.day_start_time.replace('24:00', '00:00')
        night_start_str = self.night_start_time.replace('24:00', '00:00')

        day_start = datetime.strptime(day_start_str, '%H:%M').time()
        night_start = datetime.strptime(night_start_str, '%H:%M').time()
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
        detailed_df['movement_threshold'] = 3.0  # 不動判定の閾値（3ピクセル）
        # NaTを含む行は昼夜判定をスキップ
        detailed_df['is_day'] = detailed_df['datetime'].apply(
            lambda x: self._is_daytime(x) if pd.notna(x) else None
        )
        raw_csv_path = os.path.join(output_dir, 'detailed_immobility_analysis.csv')
        detailed_df.to_csv(raw_csv_path, index=False)

        # 2. 時間集約データのCSV
        if self.processed_df is not None:
            aggregated_csv_path = os.path.join(output_dir, 'aggregated_immobility_analysis.csv')
            processed_with_time = self.processed_df.copy()
            # NaTを含む行は昼夜判定をスキップ
            processed_with_time['is_day'] = processed_with_time['datetime'].apply(
                lambda x: self._is_daytime(x) if pd.notna(x) else None
            )
            processed_with_time['immobility_percentage'] = processed_with_time['immobility_ratio'] * 100
            processed_with_time.to_csv(aggregated_csv_path, index=False)

            # 3. 昼夜別の統計サマリーCSV（NaTを除外してから計算）
            valid_data = processed_with_time.dropna(subset=['datetime', 'is_day'])
            summary_csv_path = os.path.join(output_dir, 'day_night_summary.csv')
            if len(valid_data) > 0:
                summary_stats = self._calculate_day_night_stats(valid_data)
                summary_stats.to_csv(summary_csv_path, index=False)

            print("詳細CSVファイルを保存しました:")
            print(f"  - フレームごとデータ: {raw_csv_path}")
            print(f"  - 時間集約データ: {aggregated_csv_path}")
            if len(valid_data) > 0:
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
            # 24:00を00:00に変換
            day_start_str = self.day_start_time.replace('24:00', '00:00')
            night_start_str = self.night_start_time.replace('24:00', '00:00')

            day_start = datetime.strptime(day_start_str, '%H:%M').time()
            night_start = datetime.strptime(night_start_str, '%H:%M').time()

            # NaTを除外してから日付を取得
            valid_dates = self.processed_df.dropna(subset=['datetime'])
            if len(valid_dates) == 0:
                return
            unique_dates = valid_dates['datetime'].dt.date.unique()
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
        # NaTを含む行を除外
        plot_df = self.processed_df.dropna(subset=['datetime'])
        x = plot_df['datetime']
        plt.plot(x, plot_df['total_movement'], 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, plot_df['movement_ma'], 'r-', linewidth=2, label='Moving Average (window=1)')
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
        # NaTを含む行を除外
        plot_df = self.processed_df.dropna(subset=['datetime'])
        x = plot_df['datetime']
        plt.plot(x, plot_df['immobility_ratio'] * 100, 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, plot_df['immobility_ma'] * 100, 'g-', linewidth=2, label='Moving Average (window=1)')
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
        # NaTを含む行を除外
        plot_df = self.processed_df.dropna(subset=['datetime'])
        x = plot_df['datetime']
        plt.plot(x, plot_df['mean_body_length'], 'o-', alpha=0.6, label='Raw Data', markersize=3)
        plt.plot(x, plot_df['body_length_ma'], 'purple', linewidth=2, label='Moving Average (window=1)')
        plt.errorbar(x, plot_df['mean_body_length'],
                     yerr=plot_df['std_body_length'],
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
        # NaTを含む行を除外
        plot_df = self.processed_df.dropna(subset=['datetime'])
        x = plot_df['datetime']
        # 各サブプロットに夜間背景を追加
        for ax in axes:
            self._add_night_background(ax)
        # 1. 移動量
        axes[0].plot(x, plot_df['total_movement'], 'o-', alpha=0.6, markersize=2)
        axes[0].plot(x, plot_df['movement_ma'], 'r-', linewidth=2)
        axes[0].set_ylabel(f'Movement\n({self.time_interval}min intervals)')
        axes[0].set_title('Animal Behavior Analysis')
        axes[0].grid(True, alpha=0.3)
        # 2. 不動割合
        axes[1].plot(x, plot_df['immobility_ratio'] * 100, 'o-', alpha=0.6, markersize=2)
        axes[1].plot(x, plot_df['immobility_ma'] * 100, 'g-', linewidth=2)
        axes[1].set_ylabel('Immobility Ratio (%)')
        axes[1].grid(True, alpha=0.3)
        # 3. 体長
        axes[2].plot(x, plot_df['mean_body_length'], 'o-', alpha=0.6, markersize=2)
        axes[2].plot(x, plot_df['body_length_ma'], 'purple', linewidth=2)
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
    measurement_start_time = '09:00:00'
    measurement_date_str = None
    config_path = os.path.join(csv_dir, 'time_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                time_config = json.load(f)
                day_start_time = time_config.get('day_start_time', day_start_time)
                night_start_time = time_config.get('night_start_time', night_start_time)
                measurement_start_time = time_config.get('measurement_start_time', '09:00:00')
                measurement_date_str = time_config.get('measurement_date', None)
            print(f"時間設定を読み込みました: 昼 {day_start_time}, 夜 {night_start_time}, 測定開始 {measurement_start_time}")
            if measurement_date_str:
                print(f"測定日付: {measurement_date_str}")
        except Exception as e:
            print(f"時間設定の読み込みに失敗しました: {e}, デフォルト値を使用します")
    else:
        print("時間設定ファイルが見つかりません。デフォルト値を使用します")
        measurement_start_time = '09:00:00'

    # 測定日付を変換
    measurement_date = None
    if measurement_date_str:
        try:
            from datetime import datetime
            measurement_date = datetime.strptime(measurement_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"日付の形式が正しくありません: {measurement_date_str}")
            measurement_date = None

    # 解析器を初期化（10分間隔で解析、時間設定を適用）
    analyzer = BehaviorAnalyzer(csv_path, time_interval_minutes=10,
                                day_start_time=day_start_time,
                                night_start_time=night_start_time,
                                measurement_start_time=measurement_start_time,
                                measurement_date=measurement_date)

    # データ読み込みと前処理
    if not analyzer.load_data():
        return

    # 移動量と不動性を計算
    analyzer.calculate_movement()
    analyzer.calculate_immobility_ratio()

    # 時間ビンごとに集約
    analyzer.aggregate_by_time()

    # 移動平均を適用（10分間隔の移動平均）
    analyzer.apply_moving_average(window=1)

    # グラフ作成（figuresフォルダに保存）
    analyzer.create_plots(output_dir)

    # 詳細CSV/サマリーレポート生成
    analyzer.save_detailed_csv(output_dir)
    analyzer.generate_summary_report(output_dir)


if __name__ == "__main__":
    main()
