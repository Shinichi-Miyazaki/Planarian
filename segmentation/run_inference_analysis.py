"""
セグメンテーション推論と行動解析の統合スクリプト

画像フォルダとモデルパスを指定すると、以下を実行:
1. セグメンテーション推論（CPU）
2. CSV生成
3. 行動解析
4. グラフ・統計レポート出力

behavior_analysis.pyと同様の結果を出力
"""

import os
import sys
import argparse
import json
from datetime import datetime, date

# 親ディレクトリをパスに追加（behavior_analysisをインポートするため）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 先にbehavior_analysisをインポート（pandasなどの依存関係が含まれている）
from behavior_analysis import BehaviorAnalyzer

# 次にsegmentationモジュールをインポート
import config
from inference import inference


def run_inference_and_analysis(
    images_dir,
    output_dir,
    model_path=None,
    create_video=True,
    time_interval_minutes=10,
    day_start_time='07:00',
    night_start_time='19:00',
    measurement_start_time='09:00:00',
    measurement_date=None
):
    """
    セグメンテーション推論と行動解析を統合実行

    Args:
        images_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        model_path: モデルファイルパス（Noneの場合はBEST_MODEL_PATH使用）
        create_video: 動画を作成するか
        time_interval_minutes: 時間間隔（分）
        day_start_time: 昼の開始時間 (HH:MM形式)
        night_start_time: 夜の開始時間 (HH:MM形式)
        measurement_start_time: 測定開始時間 (HH:MM:SS形式)
        measurement_date: 測定開始日付 (datetime.date オブジェクト、Noneの場合は今日)
    """
    print(f"\n{'='*70}")
    print(f"  セグメンテーション推論 + 行動解析パイプライン")
    print(f"{'='*70}\n")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 時間設定をtime_config.jsonとして保存
    time_config = {
        'day_start_time': day_start_time,
        'night_start_time': night_start_time,
        'measurement_start_time': measurement_start_time,
        'measurement_date': measurement_date.strftime('%Y-%m-%d') if measurement_date else None
    }
    config_path = os.path.join(output_dir, 'time_config.json')
    with open(config_path, 'w') as f:
        json.dump(time_config, f, indent=2)
    print(f"時間設定を保存: {config_path}")
    print(f"  - 昼: {day_start_time} - {night_start_time}")
    print(f"  - 測定開始時刻: {measurement_start_time}")
    if measurement_date:
        print(f"  - 測定日付: {measurement_date.strftime('%Y-%m-%d')}")
    print()

    # ステップ1: セグメンテーション推論
    print(f"{'='*70}")
    print(f"  ステップ1: セグメンテーション推論")
    print(f"{'='*70}\n")

    df = inference(
        images_dir=images_dir,
        output_dir=output_dir,
        model_path=model_path,
        create_video=create_video
    )

    csv_path = os.path.join(output_dir, 'analysis_results.csv')

    if not os.path.exists(csv_path):
        print(f"エラー: CSVファイルが生成されませんでした: {csv_path}")
        return

    # ステップ2: 行動解析
    print(f"\n{'='*70}")
    print(f"  ステップ2: 行動解析")
    print(f"{'='*70}\n")

    # BehaviorAnalyzerを初期化
    analyzer = BehaviorAnalyzer(
        csv_path=csv_path,
        time_interval_minutes=time_interval_minutes,
        day_start_time=day_start_time,
        night_start_time=night_start_time,
        measurement_start_time=measurement_start_time,
        measurement_date=measurement_date
    )

    # データ読み込みと前処理
    if not analyzer.load_data():
        print("エラー: データの読み込みに失敗しました")
        return

    # 移動量と不動性を計算
    print("移動量を計算中...")
    analyzer.calculate_movement()

    print("不動性を計算中...")
    analyzer.calculate_immobility_ratio()

    # 時間ビンごとに集約
    print("時間ビンごとに集約中...")
    analyzer.aggregate_by_time()

    # 移動平均を適用
    print("移動平均を適用中...")
    analyzer.apply_moving_average(window=1)

    # グラフ作成
    print("\nグラフを作成中...")
    analyzer.create_plots(output_dir)

    # 詳細CSV/サマリーレポート生成
    print("詳細CSV/サマリーレポートを生成中...")
    analyzer.save_detailed_csv(output_dir)
    analyzer.generate_summary_report(output_dir)

    print(f"\n{'='*70}")
    print(f"  全ての処理が完了しました")
    print(f"{'='*70}\n")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"  - セグメンテーション結果CSV: analysis_results.csv")
    print(f"  - 行動解析結果CSV: detailed_immobility_analysis.csv")
    print(f"  - 集約データCSV: aggregated_immobility_analysis.csv")
    print(f"  - 昼夜別統計CSV: day_night_summary.csv")
    print(f"  - グラフ: *.png")
    if create_video:
        print(f"  - セグメンテーション動画: segmentation_video.avi")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='セグメンテーション推論 + 行動解析パイプライン（CPU実行）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必須引数
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='入力画像ディレクトリ'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='出力ディレクトリ'
    )

    # オプション引数
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'モデルファイルパス（デフォルト: {config.BEST_MODEL_PATH}）'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='セグメンテーション動画を作成しない'
    )
    parser.add_argument(
        '--time-interval',
        type=int,
        default=10,
        help='時間間隔（分）'
    )
    parser.add_argument(
        '--day-start',
        type=str,
        default='07:00',
        help='昼の開始時間 (HH:MM形式)'
    )
    parser.add_argument(
        '--night-start',
        type=str,
        default='19:00',
        help='夜の開始時間 (HH:MM形式)'
    )
    parser.add_argument(
        '--measurement-start',
        type=str,
        default='09:00:00',
        help='測定開始時間 (HH:MM:SS形式)'
    )
    parser.add_argument(
        '--measurement-date',
        type=str,
        default=None,
        help='測定開始日付 (YYYY-MM-DD形式、デフォルト: 今日)'
    )

    args = parser.parse_args()

    # 測定日付を変換
    measurement_date = None
    if args.measurement_date:
        try:
            measurement_date = datetime.strptime(args.measurement_date, '%Y-%m-%d').date()
        except ValueError:
            print(f"エラー: 日付の形式が正しくありません: {args.measurement_date}")
            print("YYYY-MM-DD形式で指定してください（例: 2026-02-06）")
            return
    else:
        measurement_date = date.today()

    # 実行
    run_inference_and_analysis(
        images_dir=args.images,
        output_dir=args.output,
        model_path=args.model,
        create_video=(not args.no_video),
        time_interval_minutes=args.time_interval,
        day_start_time=args.day_start,
        night_start_time=args.night_start,
        measurement_start_time=args.measurement_start,
        measurement_date=measurement_date
    )


if __name__ == "__main__":
    main()
