"""
推論スクリプト

学習済みモデルで画像シーケンスをセグメンテーション
既存フォーマット（CSV）で出力 + 動画生成
"""

import os
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from utils import get_image_files, mask_to_contours, timestamp
from unet_model import load_model


def preprocess_image(image):
    """推論用の前処理"""
    # RGB変換
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # リサイズ
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    # 正規化
    transform = A.Compose([
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2()
    ])

    augmented = transform(image=resized)
    tensor = augmented['image'].unsqueeze(0)  # バッチ次元追加

    return tensor, (original_h, original_w)


def postprocess_mask(pred_mask, original_size, threshold=0.5):
    """予測マスクの後処理"""
    # Sigmoidを適用して0-1に変換
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = pred_mask.squeeze().cpu().numpy()

    # 二値化
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # 元のサイズにリサイズ
    binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]),
                             interpolation=cv2.INTER_NEAREST)

    return binary_mask


def inference(images_dir, output_dir, model_path=None, create_video=True):
    """
    メイン推論関数

    Args:
        images_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        model_path: モデルファイルパス（Noneの場合はBEST_MODEL_PATH使用）
        create_video: 動画を作成するか
    """
    print(f"\n{'='*60}")
    print(f"  セグメンテーション推論")
    print(f"{'='*60}\n")
    print(f"[{timestamp()}] 初期化中...")

    # モデルパス
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # デバイス
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # モデル読み込み
    print(f"[{timestamp()}] モデル読み込み中...")
    model = load_model(
        model_path=model_path,
        encoder_name=config.ENCODER_NAME,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        device=device
    )

    # 画像ファイル取得
    image_files = get_image_files(images_dir)
    if len(image_files) == 0:
        raise ValueError(f"画像が見つかりません: {images_dir}")

    print(f"画像数: {len(image_files)}")

    # 動画ライター初期化
    video_writer = None
    if create_video:
        first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
        h, w = first_img.shape[:2]
        video_path = os.path.join(output_dir, 'segmentation_video.avi')
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        video_writer = cv2.VideoWriter(video_path, fourcc, config.VIDEO_FPS, (w, h))
        print(f"動画出力: {video_path}")

    # 推論ループ
    results_list = []

    print(f"\n[{timestamp()}] 推論開始\n")

    with torch.no_grad():
        for i, img_file in enumerate(tqdm(image_files, desc='Processing')):
            img_path = os.path.join(images_dir, img_file)

            # 画像読み込み
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告: 画像を読み込めません: {img_file}")
                continue

            # 前処理
            input_tensor, original_size = preprocess_image(image)
            input_tensor = input_tensor.to(device)

            # 推論
            output = model(input_tensor)

            # 後処理
            mask = postprocess_mask(output, original_size, config.CONFIDENCE_THRESHOLD)

            # 輪郭抽出と形状特徴計算
            contour_results = mask_to_contours(
                mask,
                min_area=config.MIN_CONTOUR_AREA,
                max_area=config.MAX_CONTOUR_AREA
            )

            # 結果を記録（最も中心に近い個体を選択）
            if len(contour_results) > 0:
                # 中心に最も近い個体を選択
                img_h, img_w = mask.shape
                center_x, center_y = img_w / 2, img_h / 2

                min_dist = float('inf')
                best_result = None

                for result in contour_results:
                    dist = np.sqrt((result['centroid_x'] - center_x)**2 +
                                 (result['centroid_y'] - center_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_result = result

                # CSV用の結果
                results_list.append({
                    'filename': img_file,
                    'centroid_x': best_result['centroid_x'],
                    'centroid_y': best_result['centroid_y'],
                    'major_axis': best_result['major_axis'],
                    'minor_axis': best_result['minor_axis'],
                    'circularity': best_result['circularity'],
                    'area': best_result['area']
                })

                # 動画フレーム作成
                if video_writer is not None:
                    frame = image.copy()

                    # 輪郭描画
                    for result in contour_results:
                        cv2.drawContours(frame, [result['contour']], -1, (0, 255, 0), 2)

                    # 重心描画
                    cx = int(best_result['centroid_x'])
                    cy = int(best_result['centroid_y'])
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # 面積表示
                    text = f"Area: {best_result['area']:.0f}"
                    cv2.putText(frame, text, (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # フレーム番号
                    cv2.putText(frame, f"Frame: {i+1}/{len(image_files)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    video_writer.write(frame)
            else:
                # 検出なし
                results_list.append({
                    'filename': img_file,
                    'centroid_x': np.nan,
                    'centroid_y': np.nan,
                    'major_axis': np.nan,
                    'minor_axis': np.nan,
                    'circularity': np.nan,
                    'area': 0
                })

                if video_writer is not None:
                    # 検出なしフレーム
                    frame = image.copy()
                    cv2.putText(frame, "No detection", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Frame: {i+1}/{len(image_files)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    video_writer.write(frame)

    # 動画ライターを閉じる
    if video_writer is not None:
        video_writer.release()
        print(f"\n動画を保存しました: {video_path}")

    # CSV保存
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(output_dir, 'analysis_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  推論完了")
    print(f"{'='*60}\n")
    print(f"結果CSV: {csv_path}")
    print(f"検出成功: {df['area'].notna().sum()} / {len(df)} フレーム")
    print(f"検出率: {df['area'].notna().sum() / len(df) * 100:.1f}%\n")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='セグメンテーション推論')
    parser.add_argument('--images', type=str, required=True, help='入力画像ディレクトリ')
    parser.add_argument('--output', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--model', type=str, default=None, help='モデルファイルパス')
    parser.add_argument('--no-video', action='store_true', help='動画を作成しない')

    args = parser.parse_args()

    inference(
        images_dir=args.images,
        output_dir=args.output,
        model_path=args.model,
        create_video=(not args.no_video)
    )
