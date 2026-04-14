"""
バッチ解析後の複数日データ統合解析モジュール

複数実験日のデータを統合して以下を実施:
1. 明期開始(12:00:00)基準で相対時刻に変換
2. z-score正規化による標準化
3. 時系列プロット（移動量、不動性割合、体長）
4. 明期 vs 暗期のjitter + boxplotによる比較（対応のあるt検定付き）
5. R解析用の統合CSVを出力
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
from scipy.ndimage import uniform_filter1d
from scipy import stats

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

                # デバッグ: カラムと最初の数行を表示
                print(f"  CSVカラム: {df.columns.tolist()}")
                print(f"  データ行数: {len(df)}")

                # time_intervalを数値型として取得
                time_interval = float(result.get('time_interval', 10))
                print(f"  time_interval: {time_interval}分")

                # time_binが文字列のInterval形式の場合、数値インデックスに変換
                if 'time_bin' in df.columns:
                    # 最初の値を確認
                    first_value = df['time_bin'].iloc[0] if len(df) > 0 else None
                    print(f"  time_bin の最初の値: {first_value} (型: {type(first_value)})")

                    # 文字列で、かつ '(' が含まれている場合（Interval型の文字列表現）
                    if isinstance(first_value, str) and '(' in str(first_value):
                        print(f"  → time_binをInterval形式から数値インデックスに変換します")
                        df['time_bin'] = list(range(len(df)))
                    else:
                        # それ以外の場合も数値型に変換
                        df['time_bin'] = pd.to_numeric(df['time_bin'], errors='coerce')

                # 数値列を明示的に数値型に変換（文字列として読み込まれている場合の対策）
                numeric_columns = ['time_bin', 'total_movement', 'mean_movement', 'immobility_ratio', 'mean_body_length']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # NaN値をチェック（time_bin, total_movement, immobility_ratio, mean_body_lengthは必須）
                required_columns = ['time_bin', 'total_movement', 'immobility_ratio', 'mean_body_length']
                nan_counts = df[required_columns].isnull().sum()
                if nan_counts.sum() > 0:
                    print(f"  ⚠️ 警告: 数値変換できないデータが含まれています")
                    for col, count in nan_counts.items():
                        if count > 0:
                            print(f"    - {col}: {count}件のNaN")
                    # NaN行を削除
                    df = df.dropna(subset=required_columns)
                    if len(df) == 0:
                        print(f"  ✗ エラー: 有効なデータがありません\n")
                        continue
                    print(f"  → NaN行を削除後: {len(df)}行")

                # time_binが数値型であることを再確認
                if df['time_bin'].dtype == 'object':
                    print(f"  ⚠️ 警告: time_binがまだ文字列型です。強制的に数値に変換します")
                    df['time_bin'] = pd.to_numeric(df['time_bin'], errors='coerce')
                    df = df.dropna(subset=['time_bin'])
                    if len(df) == 0:
                        print(f"  ✗ エラー: time_binの数値化に失敗\n")
                        continue

                print(f"  time_binの型確認: {df['time_bin'].dtype}")

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
                    df['hours_from_light_start'] = df['time_bin'] * (time_interval / 60.0) - time_offset

                    print(f"  データポイント数: {len(df)}")
                    print(f"  時刻範囲: {df['hours_from_light_start'].min():.2f}h ～ {df['hours_from_light_start'].max():.2f}h (明期開始基準)")

                else:
                    # 測定開始が12:00:00以降の場合（例: 13:00:00開始）
                    # 12:00:00から測定開始までの期間は欠損値
                    time_offset_hours = (start_datetime - light_start_datetime).total_seconds() / 3600.0

                    # 各データポイントの相対時刻を計算
                    df['hours_from_light_start'] = df['time_bin'] * (time_interval / 60.0) + time_offset_hours

                    print(f"  ⚠️ 測定開始が明期開始より遅い（{time_offset_hours:.2f}時間後）")
                    print(f"  0.00h ～ {time_offset_hours:.2f}h は欠損値として扱われます")
                    print(f"  データポイント数: {len(df)}")
                    print(f"  時刻範囲: {df['hours_from_light_start'].min():.2f}h ～ {df['hours_from_light_start'].max():.2f}h")

                # 実験ID（日付）を追加
                df['experiment_id'] = start_date
                df['experiment_name'] = dir_name

                # 明期・暗期の判定
                # 明期: 12:00:00 - 24:00:00（相対時刻 0h - 12h）
                # 暗期: 24:00:00 - 12:00:00（相対時刻 12h - 24h）
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
        Min-Max正規化を適用（個体ごと、0-1の範囲）

        理由:
        - 解釈の容易さ: 0=最小値、1=最大値として直感的
        - 外れ値への頑健性: z-scoreより外れ値の影響を受けにくい
        - 生物学的妥当性: 各個体内での相対的な活動レベルの変化を見るのに適している

        正規化式: normalized = (x - min) / (max - min)
        各個体内でmin=0、max=1に標準化
        """
        print("="*70)
        print("  Min-Max正規化処理（個体ごと、0-1範囲）")
        print("="*70 + "\n")

        if self.combined_data is None:
            raise ValueError("先にload_and_combine_data()を実行してください")

        self.normalized_data = self.combined_data.copy()

        # 正規化対象の指標
        metrics = ['total_movement', 'immobility_ratio', 'mean_body_length']

        # 個体ごとにmin-max正規化
        for exp_id in self.normalized_data['experiment_id'].unique():
            exp_mask = self.normalized_data['experiment_id'] == exp_id
            exp_name = self.normalized_data.loc[exp_mask, 'experiment_name'].iloc[0]

            print(f"{exp_name} ({exp_id}):")

            for metric in metrics:
                exp_data = self.normalized_data.loc[exp_mask, metric]
                min_val = exp_data.min()
                max_val = exp_data.max()

                range_val = max_val - min_val

                if range_val > 0:
                    self.normalized_data.loc[exp_mask, f'{metric}_normalized'] = (
                        (exp_data - min_val) / range_val
                    )
                    print(f"  {metric}: 最小値={min_val:.4f}, 最大値={max_val:.4f}, 範囲={range_val:.4f}")
                else:
                    print(f"  ⚠️ {metric}: 範囲=0（全て同じ値、正規化スキップ）")
                    self.normalized_data.loc[exp_mask, f'{metric}_normalized'] = 0.5  # 中央値に設定

            print()

        print("✓ 個体ごとのMin-Max正規化完了（0-1範囲）\n")

        return self.normalized_data

    def plot_timeseries(self, smoothing_window=10, time_bin_size=1.0, mean_smooth_size=5):
        """
        時系列プロット（移動量、不動性割合、体長）- 3列レイアウトで出力

        Parameters:
        -----------
        smoothing_window : int
            個別データの移動平均ウィンドウサイズ（データポイント数）
        time_bin_size : float
            時間ビンのサイズ（時間単位）
        mean_smooth_size : int
            Mean/SEMの追加スムージングサイズ（データポイント数）

        - 12:00から次の12:00まで（24時間）のデータのみ表示
        - 3つの指標を横3列に配置
        - 縦軸: 0-1に固定（Min-Max正規化）
        - 個別データは灰色細線、平均は黒太線、SEMは半透明灰色shade
        - スムージング適用（移動平均）
        """
        print("="*70)
        print("  時系列プロット作成（3列レイアウト）")
        print(f"  スムージング設定: 個別={smoothing_window}pt, ビン={time_bin_size}hr, Mean={mean_smooth_size}pt")
        print("="*70 + "\n")

        if self.normalized_data is None:
            raise ValueError("先にnormalize_data()を実行してください")

        # 時間範囲を12:00-12:00（24時間）に制限
        plot_data = self.normalized_data[
            (self.normalized_data['hours_from_light_start'] >= 0) &
            (self.normalized_data['hours_from_light_start'] <= 24)
        ].copy()

        if len(plot_data) == 0:
            print("⚠️ 警告: 0-24時間の範囲にデータがありません\n")
            return

        print(f"プロット範囲: 0-24時間（{len(plot_data)}データポイント）")

        # 3つの指標を個別にプロット
        metrics_info = [
            ('total_movement_normalized', 'Movement (Normalized 0-1)', 'movement'),
            ('immobility_ratio_normalized', 'Immobility Ratio (Normalized 0-1)', 'immobility'),
            ('mean_body_length_normalized', 'Body Length (Normalized 0-1)', 'body_length')
        ]

        # 実験リストを取得
        experiments = plot_data['experiment_id'].unique()

        # 3列レイアウトで作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (metric, ylabel, filename) in enumerate(metrics_info):
            ax = axes[idx]

            # 個別実験のプロット（灰色細線）
            for exp_id in experiments:
                exp_data = plot_data[plot_data['experiment_id'] == exp_id].sort_values('hours_from_light_start').copy()
                exp_name = exp_data['experiment_name'].iloc[0]

                # スムージング適用（移動平均）
                exp_data[f'{metric}_smoothed'] = exp_data[metric].rolling(
                    window=smoothing_window,
                    min_periods=1,
                    center=True
                ).mean()

                ax.plot(
                    exp_data['hours_from_light_start'],
                    exp_data[f'{metric}_smoothed'],
                    linewidth=0.8,
                    alpha=0.4,
                    color='#808080',  # gray
                    zorder=5
                )

            # Mean ± SEMの計算とプロット
            # 時間をtime_bin_size単位に丸めてグループ化（スムージング効果）
            plot_data_temp = plot_data.copy()
            plot_data_temp['hours_rounded'] = (plot_data_temp['hours_from_light_start'] / time_bin_size).round() * time_bin_size

            grouped = plot_data_temp.groupby('hours_rounded').agg({
                metric: ['mean', 'std', 'count']
            }).reset_index()

            # カラム名を整理
            grouped.columns = ['hours', 'mean_value', 'std_value', 'count']

            # 時間順にソート
            grouped = grouped.sort_values('hours')

            # SEM計算: std / sqrt(n)
            grouped['sem'] = grouped['std_value'] / (grouped['count'] ** 0.5)

            # 欠損値を処理
            grouped = grouped.dropna(subset=['mean_value', 'sem'])

            time_bins = grouped['hours'].values
            mean_values = grouped['mean_value'].values
            sem_values = grouped['sem'].values

            # さらに移動平均でスムージング
            mean_values_smoothed = uniform_filter1d(mean_values, size=mean_smooth_size, mode='nearest')
            sem_values_smoothed = uniform_filter1d(sem_values, size=mean_smooth_size, mode='nearest')

            # SEMリボン（半透明灰色の網掛け）
            ax.fill_between(
                time_bins,
                mean_values_smoothed - sem_values_smoothed,
                mean_values_smoothed + sem_values_smoothed,
                alpha=0.25,
                color='#808080',  # gray
                zorder=8,
                label='Mean ± SEM' if idx == 0 else None
            )

            # Mean（黒太線）
            ax.plot(
                time_bins,
                mean_values_smoothed,
                linewidth=2.5,
                color='#000000',  # black
                zorder=10,
                label='Mean' if idx == 0 else None
            )

            # 明期・暗期の背景
            # 明期: 0-12h（白）
            # 暗期: 12-24h（グレー）
            ax.axvspan(12, 24, alpha=0.12, color='gray', zorder=0)

            # 縦軸を0-1に固定
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_xlabel('Hours from Light Start (12:00)', fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='both')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 6))

            # 0.5の参照線を追加（中央値）
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'timeseries_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 時系列プロット保存（3列レイアウト）: {output_path}")
        print()

    def plot_light_dark_comparison(self):
        """
        明期 vs 暗期の比較プロット（jitter + boxplot）- 3列レイアウトで出力

        - 1個体につき1データ点: 各個体の明期・暗期における平均値を使用
        - 3つの指標を横3列に配置
        - 縦軸: 0-1に固定（Min-Max正規化）
        - 対応のあるt検定を実施してp値を表示
        """
        print("="*70)
        print("  明期 vs 暗期 比較プロット作成（3列レイアウト）")
        print("="*70 + "\n")

        if self.normalized_data is None:
            raise ValueError("先にnormalize_data()を実行してください")

        # 時間範囲を12:00-12:00（24時間）に制限
        plot_data = self.normalized_data[
            (self.normalized_data['hours_from_light_start'] >= 0) &
            (self.normalized_data['hours_from_light_start'] <= 24)
        ].copy()

        if len(plot_data) == 0:
            print("⚠️ 警告: 0-24時間の範囲にデータがありません\n")
            return

        # 各個体・期間ごとの平均値を計算
        summary_data = plot_data.groupby(['experiment_id', 'experiment_name', 'period']).agg({
            'total_movement_normalized': 'mean',
            'immobility_ratio_normalized': 'mean',
            'mean_body_length_normalized': 'mean'
        }).reset_index()

        print(f"個体数: {summary_data['experiment_id'].nunique()}")
        print(f"データポイント数（明期+暗期）: {len(summary_data)}\n")

        # 3つの指標を3列レイアウトでプロット
        metrics_info = [
            ('total_movement_normalized', 'Movement\n(Normalized 0-1)', 'movement', '#DC143C'),  # red
            ('immobility_ratio_normalized', 'Immobility Ratio\n(Normalized 0-1)', 'immobility', '#4169E1'),  # blue
            ('mean_body_length_normalized', 'Body Length\n(Normalized 0-1)', 'body_length', '#228B22')  # green
        ]

        # 3列レイアウト（高さ控えめ）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (metric, ylabel, filename, box_color_light) in enumerate(metrics_info):
            ax = axes[idx]

            # Boxplot
            sns.boxplot(
                data=summary_data,
                x='period',
                y=metric,
                order=['Light', 'Dark'],
                ax=ax,
                width=0.6,
                palette={'Light': '#FFD700', 'Dark': '#4169E1'},
                linewidth=2
            )

            # Jitter plot（個別データポイント：1個体1点）
            sns.stripplot(
                data=summary_data,
                x='period',
                y=metric,
                order=['Light', 'Dark'],
                ax=ax,
                size=7,
                alpha=0.7,
                color='black',
                jitter=True
            )

            # 対応のあるt検定（Paired t-test）
            light_data = summary_data[summary_data['period'] == 'Light'][metric].values
            dark_data = summary_data[summary_data['period'] == 'Dark'][metric].values

            # 同じ個体数かチェック
            if len(light_data) == len(dark_data):
                t_stat, p_value = stats.ttest_rel(light_data, dark_data)

                # p値の表示形式を決定
                if p_value < 0.001:
                    p_text = f'p < 0.001'
                elif p_value < 0.01:
                    p_text = f'p = {p_value:.3f}'
                else:
                    p_text = f'p = {p_value:.3f}'

                # 有意性の表記
                if p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                elif p_value < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'

                # p値をプロット上部に表示
                y_max = max(light_data.max(), dark_data.max())
                y_pos = y_max + 0.1

                # 線を引く
                ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1.5)

                # p値とt統計量を表示
                ax.text(0.5, y_pos + 0.02, f'{sig_text}\n{p_text}\nt={t_stat:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

                print(f"  {metric}: Paired t-test")
                print(f"    Light (n={len(light_data)}): mean={light_data.mean():.4f}, std={light_data.std():.4f}")
                print(f"    Dark  (n={len(dark_data)}):  mean={dark_data.mean():.4f}, std={dark_data.std():.4f}")
                print(f"    t = {t_stat:.4f}, p = {p_value:.4f} {sig_text}\n")
            else:
                print(f"  ⚠️ {metric}: データ数が異なるため対応のあるt検定をスキップ")
                print(f"    Light: n={len(light_data)}, Dark: n={len(dark_data)}\n")

            # 0.5の参照線を追加（中央値）
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # 縦軸を0-1に固定
            ax.set_ylim(-0.05, 1.15)
            ax.set_xlabel('Period', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(ylabel.replace('\n', ' '), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'light_dark_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 明期 vs 暗期比較プロット保存（3列レイアウト）: {output_path}")
        print()

        return summary_data

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

        # 時間範囲を12:00-12:00（24時間）に制限
        output_data = self.normalized_data[
            (self.normalized_data['hours_from_light_start'] >= 0) &
            (self.normalized_data['hours_from_light_start'] <= 24)
        ].copy()

        # 1. 全データポイントのCSV（時系列解析用）
        output_columns = [
            'experiment_id',
            'experiment_name',
            'time_bin',
            'hours_from_light_start',
            'period',
            'total_movement',
            'total_movement_normalized',
            'immobility_ratio',
            'immobility_ratio_normalized',
            'mean_body_length',
            'mean_body_length_normalized',
            'mean_movement',
            'std_movement',
            'std_body_length'
        ]

        # 利用可能なカラムのみを選択
        available_columns = [col for col in output_columns if col in output_data.columns]
        output_df = output_data[available_columns].copy()

        csv_path = os.path.join(self.output_dir, 'combined_data_for_R.csv')
        output_df.to_csv(csv_path, index=False)

        print(f"✓ 統合CSV保存（全時系列データ）: {csv_path}")
        print(f"  行数: {len(output_df)}")
        print(f"  列��: {len(available_columns)}")
        print(f"  実験数: {output_df['experiment_id'].nunique()}\n")

        # 2. 個体ごとの明期・暗期平均値CSV（統計解析・ボックスプロット用）
        # 利用可能なカラムのみを集計
        agg_dict = {'total_movement': 'mean', 'immobility_ratio': 'mean', 'mean_body_length': 'mean'}

        # normalized列があれば追加
        if 'total_movement_normalized' in output_data.columns:
            agg_dict['total_movement_normalized'] = 'mean'
        if 'immobility_ratio_normalized' in output_data.columns:
            agg_dict['immobility_ratio_normalized'] = 'mean'
        if 'mean_body_length_normalized' in output_data.columns:
            agg_dict['mean_body_length_normalized'] = 'mean'

        summary_data = output_data.groupby(['experiment_id', 'experiment_name', 'period']).agg(agg_dict).reset_index()

        summary_csv_path = os.path.join(self.output_dir, 'individual_period_averages.csv')
        summary_data.to_csv(summary_csv_path, index=False)

        print(f"✓ 個体別平均値CSV保存: {summary_csv_path}")
        print(f"  個体数: {summary_data['experiment_id'].nunique()}")
        print(f"  データポイント数: {len(summary_data)}（個体 × 明期/暗期）\n")

        # サマリー統計を出力
        summary_path = os.path.join(self.output_dir, 'summary_statistics.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("  バッチ統合解析 - サマリー統計\n")
            f.write("="*70 + "\n\n")

            f.write(f"実験数: {output_df['experiment_id'].nunique()}\n")
            f.write(f"総データポイント数（時系列）: {len(output_df)}\n")
            f.write(f"個体別平均データポイント数: {len(summary_data)}\n\n")

            f.write("実験リスト:\n")
            for exp_id in output_df['experiment_id'].unique():
                exp_name = output_df[output_df['experiment_id'] == exp_id]['experiment_name'].iloc[0]
                count = len(output_df[output_df['experiment_id'] == exp_id])
                f.write(f"  - {exp_name} ({exp_id}): {count} データポイント\n")
            f.write("\n")

            f.write("明期・暗期別データポイント数（時系列）:\n")
            for period in ['Light', 'Dark']:
                count = len(output_df[output_df['period'] == period])
                f.write(f"  - {period}: {count} データポイント\n")
            f.write("\n")

            f.write("="*70 + "\n")
            f.write("  個体別平均値の統計（明期 vs 暗期比較用）\n")
            f.write("="*70 + "\n\n")

            for period in ['Light', 'Dark']:
                period_data = summary_data[summary_data['period'] == period]
                f.write(f"{period} Period:\n")
                f.write(f"  個体数: {len(period_data)}\n")
                for metric in ['total_movement_zscore', 'immobility_ratio_zscore', 'mean_body_length_zscore']:
                    if metric in period_data.columns:
                        f.write(f"  {metric}: 平均={period_data[metric].mean():.4f}, "
                                f"標準偏差={period_data[metric].std():.4f}\n")
                f.write("\n")

            # 対応のあるt検定の結果を追加
            f.write("="*70 + "\n")
            f.write("  対応のあるt検定（Paired t-test: Light vs Dark）\n")
            f.write("="*70 + "\n\n")

            for metric in ['total_movement', 'immobility_ratio', 'mean_body_length']:
                normalized_col = f'{metric}_normalized'
                if normalized_col in summary_data.columns:
                    light_data = summary_data[summary_data['period'] == 'Light'][normalized_col].values
                    dark_data = summary_data[summary_data['period'] == 'Dark'][normalized_col].values

                    if len(light_data) == len(dark_data) and len(light_data) > 0:
                        t_stat, p_value = stats.ttest_rel(light_data, dark_data)

                        # 有意性判定
                        if p_value < 0.001:
                            sig = '***'
                        elif p_value < 0.01:
                            sig = '**'
                        elif p_value < 0.05:
                            sig = '*'
                        else:
                            sig = 'n.s.'

                        f.write(f"{metric}:\n")
                        f.write(f"  Light: n={len(light_data)}, mean={light_data.mean():.4f}, std={light_data.std():.4f}\n")
                        f.write(f"  Dark:  n={len(dark_data)}, mean={dark_data.mean():.4f}, std={dark_data.std():.4f}\n")
                        f.write(f"  t = {t_stat:.4f}, p = {p_value:.6f} {sig}\n\n")
                    else:
                        f.write(f"{metric}: データ数不一致（Light: {len(light_data)}, Dark: {len(dark_data)}）\n\n")

            f.write("  有意水準: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant\n\n")

            f.write("="*70 + "\n")
            f.write("  Min-Max正規化後の基本統計量（全時系列データ）\n")
            f.write("="*70 + "\n\n")

            for metric in ['total_movement', 'immobility_ratio', 'mean_body_length']:
                normalized_col = f'{metric}_normalized'
                if normalized_col in output_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  最小値: {output_df[normalized_col].min():.4f}\n")
                    f.write(f"  最大値: {output_df[normalized_col].max():.4f}\n")
                    f.write(f"  範囲: {output_df[normalized_col].max() - output_df[normalized_col].min():.4f}\n\n")


        print(f"✓ サマリー統計保存: {summary_path}\n")

        return csv_path

    def run_full_analysis(self, smoothing_window=10, time_bin_size=1.0, mean_smooth_size=5):
        """
        統合解析の全ステップを実行

        Parameters:
        -----------
        smoothing_window : int
            個別データの移動平均ウィンドウサイズ（データポイント数）
        time_bin_size : float
            時間ビンのサイズ（時間単位）
        mean_smooth_size : int
            Mean/SEMの追加スムージングサイズ（データポイント数）
        """
        print("\n" + "="*70)
        print("  バッチ統合解析開始")
        print("="*70 + "\n")

        # 1. データ統合
        self.load_and_combine_data()

        # 2. 正規化
        self.normalize_data()

        # 3. 時系列プロット
        self.plot_timeseries(
            smoothing_window=smoothing_window,
            time_bin_size=time_bin_size,
            mean_smooth_size=mean_smooth_size
        )

        # 4. 明期・暗期比較プロット
        self.plot_light_dark_comparison()

        # 5. R解析用CSV出力
        self.save_combined_csv()

        print("="*70)
        print("  バッチ統合解析完了")
        print("="*70 + "\n")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"  - timeseries_comparison.png: 時系列プロット（3列レイアウト）")
        print(f"  - light_dark_comparison.png: 明期 vs 暗期比較（3列レイアウト）")
        print(f"  - combined_data_for_R.csv: R解析用統合データ（全時系列）")
        print(f"  - individual_period_averages.csv: 個体別明期・暗期平均値")
        print(f"  - summary_statistics.txt: サマリー統計\n")


def run_batch_summary_analysis(analysis_results, output_dir,
                               smoothing_window=10,
                               time_bin_size=1.0,
                               mean_smooth_size=5):
    """
    バッチ統合解析を実行する便利関数

    Parameters:
    -----------
    analysis_results : list of dict
        各実験の解析結果リスト
    output_dir : str
        出力ディレクトリ
    smoothing_window : int, optional
        個別データの移動平均ウィンドウサイズ（データポイント数）, default=10
    time_bin_size : float, optional
        時間ビンのサイズ（時間単位）, default=1.0
    mean_smooth_size : int, optional
        Mean/SEMの追加スムージングサイズ（データポイント数）, default=5
    """
    analyzer = BatchSummaryAnalyzer(analysis_results, output_dir)
    analyzer.run_full_analysis(
        smoothing_window=smoothing_window,
        time_bin_size=time_bin_size,
        mean_smooth_size=mean_smooth_size
    )
    return analyzer

