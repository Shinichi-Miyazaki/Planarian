"""
サンプルのExcelファイルを作成するスクリプト

Excelベースのバッチ解析機能のテスト用
"""

import pandas as pd
import os


def create_sample_excel():
    """
    サンプルのExcelファイルを作成
    """
    # サンプルデータ
    data = {
        'dir_name': [
            'experiment_day1',
            'experiment_day2',
            'experiment_day3',
            'experiment_day4',
            'control_day1',
            'control_day2',
        ],
        'start_date': [
            '20260206',
            '20260207',
            '20260208',
            '20260209',
            '20260206',
            '20260207',
        ],
        'start_time': [
            '08:00:00',
            '08:15:00',
            '07:50:00',
            '08:10:00',
            '13:30:00',  # 測定開始が12:00:00以降の例
            '13:45:00',  # 測定開始が12:00:00以降の例
        ]
    }

    # DataFrameを作成
    df = pd.DataFrame(data)

    # Excelファイルとして保存
    output_path = 'sample_experiment_conditions.xlsx'
    df.to_excel(output_path, index=False, sheet_name='実験条件')

    print(f"サンプルExcelファイルを作成しました: {output_path}")
    print(f"\n内容:")
    print(df.to_string(index=False))
    print(f"\n使用方法:")
    print(f"1. このファイルをテンプレートとして使用")
    print(f"2. dir_name, start_date, start_time を実際の値に変更")
    print(f"3. GUIの「Excelファイル」から選択")
    print(f"\n注意:")
    print(f"- start_date: yyyyMMdd形式 (例: 20260210)")
    print(f"- start_time: HH:mm:ss形式 (例: 08:00:00)")


if __name__ == "__main__":
    create_sample_excel()

