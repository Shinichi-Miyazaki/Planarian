"""
バッチ解析後の複数日データ統合解析モジュール

複数実験日のデータを統合して以下を実施:
1. 明期開始(12:00:00)基準で相対時刻に変換
2. z-score正規化による標準化
3. 時系列プロット（移動量、不動性割合、体長）
4. 明期 vs 暗期のjitter + boxplotによる比較
5. R解析用の統合CSVを出力
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# カラーパレット（指定色リストから選択）
COLOR_PALETTE = [
    '#4169E1',  # blue_base
    '#DC143C',  # red_base
    '#228B22',  # green_base
    '#FFA500',  # orange_base
    '#800080',  # purple_base
    '#FF00FF',  # magenta_base
    '#808080',  # gray_base
]


class BatchSummaryAnalyzer:
    """
    バッチ解析の統合解析クラス

    複数日の実験データを統合して時系列解析と明暗期比較を実施
    """

    def __init__(self, analysis_results, output_dir):
        """
        Parameters:
        -----------
        analysis_results : list of dict
            各実験の解析結果リスト
            [{'dir_name': str, 'start_date': str(yyyyMMdd), 'start_time': str(HH:mm:ss),
              'output_dir': str, 'aggregated_csv': str}, ...]
        output_dir : str
            統合解析結果の出力ディレクトリ
        """
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        self.combined_data = None
        self.normalized_data = None

        # 明期開始時刻（固定: 12:00:00）
        self.light_period_start = '12:00:00'

        os.makedirs(output_dir, exist_ok=True)

    def load_and_combine_data(self):
        """
        各実験のaggregated CSVを読み込み、明期開始基準の相対時刻に変換して統合

        注意: 測定開始時刻が12:00:00以降の場合、12:00:00から測定開始までは欠損値として扱う
        """
        print("\n" + "="*70)
        print("  複数日データの統合処理")
        print("="*70 + "\n")

        all_data = []

        for result in self.analysis_results:
            try:
                dir_name = result['dir_name']
                start_date = result['start_date']  # yyyyMMdd形式
                start_time = result['start_time']  # HH:mm:ss形式
                aggregated_csv = result['aggregated_csv']

                print(f"読み込み中: {dir_name}")
                print(f"  測定開始: {start_date} {start_time}")

                # CSVを読み込み
                df = pd.read_csv(aggregated_csv)

                # 日付をパース
                exp_date = datetime.strptime(start_date, '%Y%m%d').date()

                # 測定開始時刻をパース
                start_time_obj = datetime.strptime(start_time, '%H:%M:%S').time()
                start_datetime = datetime.combine(exp_date, start_time_obj)

                # 明期開始時刻（12:00:00）を設定
                light_start_time_obj = datetime.strptime(self.light_period_start, '%H:%M:%S').time()
                light_start_datetime = datetime.combine(exp_date, light_start_time_obj)

                # 測定開始が12:00:00より前か後かで処理を分岐
                if start_datetime < light_start_datetime:
                    # 通常ケース: 測定開始が12:00:00より前（例: 08:00:00開始）
                    # 明期開始までの時間を負の値として計算
                    time_offset = (light_start_datetime - start_datetime).total_seconds() / 3600.0  # 時間単位

                    # 各データポイントの相対時刻を計算（測定開始からの経過時間 - オフセット）
                    df['hours_from_light_start'] = df['time_bin'] * (result.get('time_interval', 10) / 60.0) - time_offset

                    print(f"  データポイント数: {len(df)}")
                    print(f"  時刻範囲: {df['hours_from_light_start'].min():.2f}h ～ {df['hours_from_light_start'].max():.2f}h (明期開始基準)")

                else:
                    # 測定開始が12:00:00以降の場合（例: 13:00:00開始）
                    # 12:00:00から測定開始までの期間は欠損値
                    time_offset_hours = (start_datetime - light_start_datetime).total_seconds() / 3600.0

                    # 各データポイントの相対時刻を計算
                    df['hours_from_light_start'] = df['time_bin'] * (result.get('time_interval', 10) / 60.0) + time_offset_hours

                    print(f"  ⚠️ 測定開始が明期開始より遅い（{time_offset_hours:.2f}時間後）")
                    print(f"  0.00h ～ {time_offset_hours:.2f}h は欠損値として扱われます")
                    print(f"  データポイント数: {len(df)}")
                    print(f"  時刻範囲: {df['hours_from_light_start'].min():.2f}h ～ {df['hours_from_light_start'].max():.2f}h")

                # 実験ID（日付）を追加
                df['experiment_id'] = start_date
                df['experiment_name'] = dir_name

                # 明期・暗期の判定
                # 明期: 12:00:00 - 24:00:00 (0h - 12h)
                # 暗期: 00:00:00 - 12:00:00 (12h - 24h, 翌日の明期開始まで)
                df['period'] = df['hours_from_light_start'].apply(
                    lambda x: 'Light' if 0 <= (x % 24) < 12 else 'Dark'
                )

                all_data.append(df)
                print(f"  ✓ 読み込み完了\n")

            except Exception as e:
                print(f"  ✗ エラー: {e}\n")
                continue

        if not all_data:
            raise ValueError("有効なデータが1つも読み込めませんでした")

        # 全データを統合
        self.combined_data = pd.concat(all_data, ignore_index=True)

        print(f"統合完了: {len(self.analysis_results)}実験 × 合計{len(self.combined_data)}データポイント")
        print()

        return self.combined_data

    def normalize_data(self):
        """
        z-score正規化を適用

        理由: 個体差や測定日間のベースライン変動を補正し、パターンの比較を容易にする
              z-score = (x - mean) / std により、平均0、標準偏差1に標準化
              これにより異なる実験間での相対的な変動パターンを比較可能
        """
        print("="*70)
        print("  z-score正規化処理")
        print("="*70 + "\n")

        if self.combined_data is None:
            raise ValueError("先にload_and_combine_data()を実行してください")

        self.normalized_data = self.combined_data.copy()

        # 正規化対象の指標
        metrics = ['total_movement', 'immobility_ratio', 'mean_body_length']

        for metric in metrics:
            # 全区間データでz-score正規化
            mean_val = self.normalized_data[metric].mean()
            std_val = self.normalized_data[metric].std()

            if std_val > 0:
                self.normalized_data[f'{metric}_zscore'] = (
                    (self.normalized_data[metric] - mean_val) / std_val
                )
                print(f"{metric}:")
                print(f"  平均: {mean_val:.4f}, 標準偏差: {std_val:.4f}")
                print(f"  正規化後の範囲: {self.normalized_data[f'{metric}_zscore'].min():.4f} ～ "
                      f"{self.normalized_data[f'{metric}_zscore'].max():.4f}\n")
            else:
                print(f"⚠️ {metric}: 標準偏差が0のため正規化スキップ\n")
                self.normalized_data[f'{metric}_zscore'] = 0

        print("✓ 正規化完了\n")

        return self.normalized_data

    def plot_timeseries(self):
        """
        時系列プロット（移動量、不動性割合、体長）

        複数日のデータを異なる色/線種で重ねて表示
        """
        print("="*70)
        print("  時系列プロット作成")
        print("="*70 + "\n")

        if self.normalized_data is None:
            raise ValueError("先にnormalize_data()を実行してください")

        # 3つのサブプロット（移動量、不動性割合、体長）
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        metrics = [
            ('total_movement_zscore', 'Movement (z-score)', 'Total Movement'),
            ('immobility_ratio_zscore', 'Immobility Ratio (z-score)', 'Immobility Ratio'),
            ('mean_body_length_zscore', 'Body Length (z-score)', 'Body Length')
        ]

        # 実験ごとに異なる色を割り当て
        experiments = self.normalized_data['experiment_id'].unique()
        colors = COLOR_PALETTE[:len(experiments)]

        for idx, (metric, ylabel, title) in enumerate(metrics):
            ax = axes[idx]

            for exp_id, color in zip(experiments, colors):
                exp_data = self.normalized_data[self.normalized_data['experiment_id'] == exp_id]
                exp_name = exp_data['experiment_name'].iloc[0]

                # 時系列プロット
                ax.plot(
                    exp_data['hours_from_light_start'],
                    exp_data[metric],
                    marker='o',
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.7,
                    color=color,
                    label=f"{exp_name}"
                )

            # 明期・暗期の背景
            # 明期: 0-12h, 24-36h, ...
            # 暗期: 12-24h, 36-48h, ...
            x_min, x_max = ax.get_xlim()
            for start_h in range(int(x_min // 24) * 24, int(x_max) + 24, 24):
                # 暗期の背景（グレー）
                ax.axvspan(start_h + 12, start_h + 24, alpha=0.2, color='gray', zorder=0)

            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

        axes[-1].set_xlabel('Hours from Light Period Start (12:00:00)', fontsize=11)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'timeseries_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 時系列プロット保存: {output_path}\n")

    def plot_light_dark_comparison(self):
        """
        明期 vs 暗期の比較プロット（jitter + boxplot）

        各指標について明期と暗期のデータをjitter plotとboxplotで比較
        """
        print("="*70)
        print("  明期 vs 暗期 比較プロット作成")
        print("="*70 + "\n")

        if self.normalized_data is None:
            raise ValueError("先にnormalize_data()を実行してください")

        # 3つのサブプロット
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = [
            ('total_movement_zscore', 'Movement\n(z-score)'),
            ('immobility_ratio_zscore', 'Immobility Ratio\n(z-score)'),
            ('mean_body_length_zscore', 'Body Length\n(z-score)')
        ]

        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[idx]

            # Boxplot
            sns.boxplot(
                data=self.normalized_data,
                x='period',
                y=metric,
                order=['Light', 'Dark'],
                ax=ax,
                width=0.5,
                palette={'Light': '#FFD700', 'Dark': '#4169E1'},
                linewidth=1.5
            )

            # Jitter plot（個別データポイント）
            sns.stripplot(
                data=self.normalized_data,
                x='period',
                y=metric,
                order=['Light', 'Dark'],
                ax=ax,
                size=2,
                alpha=0.4,
                color='black',
                jitter=True
            )

            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Period', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(ylabel.replace('\n', ' '), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'light_dark_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 明期 vs 暗期 比較プロット保存: {output_path}\n")

    def save_combined_csv(self):
        """
        R解析用の統合CSVを保存

        全実験データを1つのCSVにまとめ、Rでさらに詳細な統計解析を実施可能にする
        """
        print("="*70)
        print("  R解析用CSV出力")
        print("="*70 + "\n")

        if self.normalized_data is None:
            raise ValueError("先にnormalize_data()を実行してください")

        # 出力用のカラムを選択・整理
        output_columns = [
            'experiment_id',
            'experiment_name',
            'time_bin',
            'hours_from_light_start',
            'period',
            'total_movement',
            'total_movement_zscore',
            'immobility_ratio',
            'immobility_ratio_zscore',
            'mean_body_length',
            'mean_body_length_zscore',
            'mean_movement',
            'std_movement',
            'std_body_length'
        ]

        # 利用可能なカラムのみを選択
        available_columns = [col for col in output_columns if col in self.normalized_data.columns]
        output_df = self.normalized_data[available_columns].copy()

        # CSVとして保存
        csv_path = os.path.join(self.output_dir, 'combined_data_for_R.csv')
        output_df.to_csv(csv_path, index=False)

        print(f"✓ 統合CSV保存: {csv_path}")
        print(f"  行数: {len(output_df)}")
        print(f"  列数: {len(available_columns)}")
        print(f"  実験数: {output_df['experiment_id'].nunique()}\n")

        # サマリー統計を出力
        summary_path = os.path.join(self.output_dir, 'summary_statistics.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("  バッチ統合解析 - サマリー統計\n")
            f.write("="*70 + "\n\n")

            f.write(f"実験数: {output_df['experiment_id'].nunique()}\n")
            f.write(f"総データポイント数: {len(output_df)}\n\n")

            f.write("実験リスト:\n")
            for exp_id in output_df['experiment_id'].unique():
                exp_name = output_df[output_df['experiment_id'] == exp_id]['experiment_name'].iloc[0]
                count = len(output_df[output_df['experiment_id'] == exp_id])
                f.write(f"  - {exp_name} ({exp_id}): {count} データポイント\n")
            f.write("\n")

            f.write("明期・暗期別データポイント数:\n")
            for period in ['Light', 'Dark']:
                count = len(output_df[output_df['period'] == period])
                f.write(f"  - {period}: {count} データポイント\n")
            f.write("\n")

            f.write("="*70 + "\n")
            f.write("  z-score正規化後の基本統計量\n")
            f.write("="*70 + "\n\n")

            for metric in ['total_movement', 'immobility_ratio', 'mean_body_length']:
                zscore_col = f'{metric}_zscore'
                if zscore_col in output_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  平均: {output_df[zscore_col].mean():.4f}\n")
                    f.write(f"  標準偏差: {output_df[zscore_col].std():.4f}\n")
                    f.write(f"  最小値: {output_df[zscore_col].min():.4f}\n")
                    f.write(f"  最大値: {output_df[zscore_col].max():.4f}\n\n")

            # 明期・暗期別の統計
            f.write("="*70 + "\n")
            f.write("  明期 vs 暗期 統計比較\n")
            f.write("="*70 + "\n\n")

            for period in ['Light', 'Dark']:
                period_data = output_df[output_df['period'] == period]
                f.write(f"{period} Period:\n")
                for metric in ['total_movement', 'immobility_ratio', 'mean_body_length']:
                    zscore_col = f'{metric}_zscore'
                    if zscore_col in period_data.columns:
                        f.write(f"  {metric}_zscore: 平均={period_data[zscore_col].mean():.4f}, "
                                f"標準偏差={period_data[zscore_col].std():.4f}\n")
                f.write("\n")

        print(f"✓ サマリー統計保存: {summary_path}\n")

        return csv_path

    def run_full_analysis(self):
        """
        統合解析の全ステップを実行
        """
        print("\n" + "="*70)
        print("  バッチ統合解析開始")
        print("="*70 + "\n")

        # 1. データ統合
        self.load_and_combine_data()

        # 2. 正規化
        self.normalize_data()

        # 3. 時系列プロット
        self.plot_timeseries()

        # 4. 明期・暗期比較プロット
        self.plot_light_dark_comparison()

        # 5. R解析用CSV出力
        self.save_combined_csv()

        print("="*70)
        print("  バッチ統合解析完了")
        print("="*70 + "\n")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"  - timeseries_comparison.png: 時系列プロット")
        print(f"  - light_dark_comparison.png: 明期 vs 暗期比較")
        print(f"  - combined_data_for_R.csv: R解析用統合データ")
        print(f"  - summary_statistics.txt: サマリー統計\n")


def run_batch_summary_analysis(analysis_results, output_dir):
    """
    バッチ統合解析を実行する便利関数

    Parameters:
    -----------
    analysis_results : list of dict
        各実験の解析結果リスト
    output_dir : str
        出力ディレクトリ
    """
    analyzer = BatchSummaryAnalyzer(analysis_results, output_dir)
    analyzer.run_full_analysis()
    return analyzer

