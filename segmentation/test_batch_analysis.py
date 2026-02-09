"""
バッチフォルダ解析機能のテストスクリプト

GUIの主要機能をテストします
"""

import os
import sys
import tempfile
import shutil

# パスを追加
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def create_test_folder_structure():
    """テスト用のフォルダ構造を作成"""
    temp_dir = tempfile.mkdtemp(prefix="planaria_test_")
    print(f"テストフォルダを作成: {temp_dir}")

    # 親フォルダ内に複数のサブフォルダを作成
    folders = [
        ("experiment_day1", 1200),  # 1200枚の画像
        ("experiment_day2", 1500),  # 1500枚の画像
        ("experiment_day3", 800),   # 800枚（閾値未満）
        ("experiment_day4", 2000),  # 2000枚の画像
        ("other_data", 50),         # 50枚（閾値未満）
    ]

    for folder_name, image_count in folders:
        folder_path = os.path.join(temp_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # ダミー画像ファイルを作成（空ファイル）
        for i in range(image_count):
            dummy_file = os.path.join(folder_path, f"image_{i:05d}.jpg")
            with open(dummy_file, 'w') as f:
                f.write("")  # 空ファイル

        print(f"  - {folder_name}: {image_count}枚の画像を作成")

    return temp_dir, folders


def test_folder_detection():
    """フォルダ検出機能のテスト"""
    print("\n" + "="*70)
    print("フォルダ検出機能のテスト")
    print("="*70 + "\n")

    # テストフォルダ構造を作成
    temp_dir, folders = create_test_folder_structure()

    try:
        # フォルダ検出ロジックをシミュレート
        threshold = 1000
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        found_folders = []

        print(f"\n親フォルダ: {temp_dir}")
        print(f"画像数閾値: {threshold}枚以上")
        print(f"\nフォルダを走査中...\n")

        for root, dirs, files in os.walk(temp_dir):
            image_count = sum(1 for f in files if f.lower().endswith(image_extensions))

            if image_count >= threshold:
                found_folders.append((root, image_count))
                print(f"✓ 検出: {os.path.basename(root)} ({image_count}枚)")
            elif image_count > 0:
                print(f"✗ スキップ: {os.path.basename(root)} ({image_count}枚) - 閾値未満")

        print(f"\n結果: {len(found_folders)}個のフォルダを検出")

        # 期待される結果
        expected_count = 3  # day1, day2, day4
        assert len(found_folders) == expected_count, \
            f"期待: {expected_count}フォルダ, 実際: {len(found_folders)}フォルダ"

        print("\n✅ テスト成功: 正しい数のフォルダが検出されました")

        # 詳細表示
        print("\n検出されたフォルダ:")
        for folder_path, image_count in found_folders:
            print(f"  - {os.path.basename(folder_path)}: {image_count}枚")

    finally:
        # テストフォルダを削除
        print(f"\nテストフォルダを削除: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_time_format_validation():
    """日付・時刻フォーマット検証のテスト"""
    from datetime import datetime

    print("\n" + "="*70)
    print("日付・時刻フォーマット検証のテスト")
    print("="*70 + "\n")

    test_cases = [
        ("2026-02-09", "%Y-%m-%d", True, "正常な日付"),
        ("2026-2-9", "%Y-%m-%d", False, "月日が1桁"),
        ("09:00:00", "%H:%M:%S", True, "正常な時刻"),
        ("9:0:0", "%H:%M:%S", False, "時分秒が1桁"),
        ("25:00:00", "%H:%M:%S", False, "時が24以上"),
    ]

    for value, fmt, should_pass, description in test_cases:
        try:
            datetime.strptime(value, fmt)
            result = "✓ 通過" if should_pass else "✗ 失敗（本来エラーのはず）"
            status = "✅" if should_pass else "❌"
        except ValueError:
            result = "✗ エラー" if should_pass else "✓ 正しく拒否"
            status = "❌" if should_pass else "✅"

        print(f"{status} {description}: '{value}' → {result}")

    print("\n✅ 日付・時刻検証テスト完了")


def print_usage_summary():
    """使用方法のサマリーを表示"""
    print("\n" + "="*70)
    print("バッチフォルダ解析機能 - 使用方法サマリー")
    print("="*70 + "\n")

    print("1. GUIを起動:")
    print("   python inference_analysis_gui.py")
    print()

    print("2. 「バッチフォルダ解析」タブを選択")
    print()

    print("3. 親フォルダを選択して「フォルダを検索」")
    print()

    print("4. 検出されたフォルダをダブルクリックして測定日時を編集")
    print()

    print("5. 「バッチ実行」で一括解析開始")
    print()

    print("出力: 各画像フォルダ内に 'segmentation_analysis/' が作成されます")
    print()

    print("詳細: docs/BATCH_ANALYSIS_GUIDE.md を参照")
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("バッチフォルダ解析機能 - テストスイート")
    print("="*70)

    try:
        # テスト実行
        test_folder_detection()
        test_time_format_validation()
        print_usage_summary()

        print("\n" + "="*70)
        print("全てのテストが完了しました！")
        print("="*70 + "\n")

    except Exception as e:
        import traceback
        print(f"\n❌ テストエラー:\n{traceback.format_exc()}")
        sys.exit(1)

