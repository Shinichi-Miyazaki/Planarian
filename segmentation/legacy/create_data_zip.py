"""
Google Colab用のデータZIPファイル作成スクリプト

segmentation/data/ フォルダを data.zip に圧縮します
ZIP内の構造: images/, labels/ （dataフォルダは含まない）
"""

import os
import zipfile
from pathlib import Path

def create_data_zip():
    """データフォルダをZIP圧縮"""

    # パス設定
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    zip_path = script_dir / 'data.zip'

    if not data_dir.exists():
        print(f"エラー: {data_dir} が見つかりません")
        return

    # ZIP作成
    print(f"データをZIP圧縮しています...")
    print(f"元フォルダ: {data_dir}")
    print(f"出力先: {zip_path}\n")

    file_count = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = Path(root) / file
                # ZIP内のパス（images/xxx.jpg, labels/xxx.png のような形式）
                # data/ フォルダを含めない
                arcname = file_path.relative_to(data_dir)
                zipf.write(file_path, arcname)
                file_count += 1

                if file_count % 10 == 0:
                    print(f"  処理中... {file_count} ファイル")

    # ファイルサイズ確認
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ ZIP作成完了！")
    print(f"  ファイル数: {file_count}")
    print(f"  ZIP サイズ: {zip_size_mb:.2f} MB")
    print(f"  保存先: {zip_path}")

    # 内容確認
    print(f"\n=== ZIP の構造 ===")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()

        # 画像とラベルをカウント
        images = [n for n in namelist if n.startswith('images/') and n.endswith(('.jpg', '.png'))]
        labels = [n for n in namelist if n.startswith('labels/') and n.endswith('.png')]

        print(f"  画像: {len(images)} 枚 (images/ フォルダ)")
        print(f"  ラベル: {len(labels)} 枚 (labels/ フォルダ)")

        # サンプル表示
        print(f"\n  最初の5ファイル:")
        for name in namelist[:5]:
            print(f"    {name}")

        if len(namelist) > 5:
            print(f"    ... 他 {len(namelist) - 5} ファイル")

    print(f"\n📌 Google Colab での使用方法:")
    print(f"   1. このZIPファイルをGoogle Colabにアップロード")
    print(f"   2. 解凍すると以下の構造になります:")
    print(f"      指定したベースディレクトリ/")
    print(f"      └─ data/")
    print(f"         ├─ images/")
    print(f"         │  ├─ 00012.jpg")
    print(f"         │  └─ ...")
    print(f"         └─ labels/")
    print(f"            ├─ 00012.png")
    print(f"            └─ ...")

if __name__ == "__main__":
    create_data_zip()
