"""
ONNX Runtime推論エンジン

RTX 5070 Ti等の新しいGPUでの推論を可能にするため、
ONNX RuntimeベースのGPU推論エンジンを提供します。

DirectML（Windows）またはCUDA（Linux/Windows）バックエンドに対応。
"""

import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

import config
from utils import get_image_files, mask_to_contours, timestamp


class ONNXInferenceEngine:
    """ONNX Runtime推論エンジン"""

    def __init__(self, model_path, device='cpu', use_directml=False):
        """
        初期化

        Args:
            model_path: ONNXモデルパス
            device: 'cpu', 'cuda', または 'directml'
            use_directml: DirectMLを使用（Windows GPU）
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntimeがインストールされていません。\n"
                "インストール: pip install onnxruntime onnxruntime-directml"
            )

        self.model_path = model_path
        self.device = device

        # セッションオプション
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # プロバイダーを選択
        providers = []

        if use_directml or device == 'directml':
            # DirectML（Windows GPU）
            try:
                providers.append('DmlExecutionProvider')
                print("DirectML (Windows GPU)を使用")
            except:
                print("⚠ DirectMLが利用できません。CPUにフォールバック")
                providers.append('CPUExecutionProvider')
        elif device == 'cuda':
            # CUDA（NVIDIA GPU）
            try:
                providers.append('CUDAExecutionProvider')
                print("CUDA (NVIDIA GPU)を使用")
            except:
                print("⚠ CUDAが利用できません。CPUにフォールバック")
                providers.append('CPUExecutionProvider')
        else:
            # CPU
            providers.append('CPUExecutionProvider')
            print("CPUを使用")

        # セッションを作成
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 入出力情報を取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ ONNXモデル読み込み完了")
        print(f"  入力: {self.input_name}")
        print(f"  出力: {self.output_name}")
        print(f"  プロバイダー: {self.session.get_providers()}")

    def preprocess_image(self, image):
        """画像の前処理"""
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
        image_float = resized.astype(np.float32) / 255.0
        mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(config.IMAGENET_STD, dtype=np.float32)
        normalized = (image_float - mean) / std

        # (H, W, C) -> (1, C, H, W)
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0).astype(np.float32)

        return batched, (original_h, original_w)

    def postprocess_mask(self, pred_mask, original_size, threshold=0.5):
        """予測マスクの後処理"""
        # Sigmoidを適用
        pred_mask = 1 / (1 + np.exp(-pred_mask))
        pred_mask = pred_mask.squeeze()

        # 二値化
        binary_mask = (pred_mask > threshold).astype(np.uint8)

        # 元のサイズにリサイズ
        binary_mask = cv2.resize(
            binary_mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return binary_mask

    def predict(self, image):
        """単一画像の推論"""
        # 前処理
        input_data, original_size = self.preprocess_image(image)

        # 推論
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        # 後処理
        mask = self.postprocess_mask(
            outputs[0],
            original_size,
            config.CONFIDENCE_THRESHOLD
        )

        return mask


def inference_onnx(
    images_dir,
    output_dir,
    model_path=None,
    create_video=True,
    device='cpu',
    use_directml=False
):
    """
    ONNX Runtimeによる推論

    Args:
        images_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        model_path: ONNXモデルパス
        create_video: 動画を作成するか
        device: 'cpu', 'cuda', 'directml'
        use_directml: DirectMLを使用（Windows GPU）
    """
    print(f"\n{'='*60}")
    print(f"  ONNX Runtime推論")
    print(f"{'='*60}\n")
    print(f"[{timestamp()}] 初期化中...")

    # モデルパス
    if model_path is None:
        model_path = config.BEST_MODEL_PATH.replace('.pth', '.onnx')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: {model_path}\n"
            f"export_onnx.pyでモデルを変換してください。"
        )

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 推論エンジンを初期化
    print(f"[{timestamp()}] ONNX Runtime初期化中...")
    engine = ONNXInferenceEngine(model_path, device=device, use_directml=use_directml)

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

    for i, img_file in enumerate(tqdm(image_files, desc='Processing')):
        img_path = os.path.join(images_dir, img_file)

        # 画像読み込み
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 画像を読み込めません: {img_file}")
            continue

        # 推論
        mask = engine.predict(image)

        # 輪郭抽出と形状特徴計算
        contour_results = mask_to_contours(
            mask,
            min_area=config.MIN_CONTOUR_AREA,
            max_area=config.MAX_CONTOUR_AREA
        )

        # 結果を記録
        if len(contour_results) > 0:
            # 中心に最も近い個体を選択
            img_h, img_w = mask.shape
            center_x, center_y = img_w / 2, img_h / 2

            min_dist = float('inf')
            best_result = None

            for result in contour_results:
                dist = np.sqrt(
                    (result['centroid_x'] - center_x)**2 +
                    (result['centroid_y'] - center_y)**2
                )
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

                for result in contour_results:
                    cv2.drawContours(frame, [result['contour']], -1, (0, 255, 0), 2)

                cx = int(best_result['centroid_x'])
                cy = int(best_result['centroid_y'])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                text = f"Area: {best_result['area']:.0f}"
                cv2.putText(frame, text, (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

    parser = argparse.ArgumentParser(description='ONNX Runtime推論')
    parser.add_argument('--images', type=str, required=True, help='入力画像ディレクトリ')
    parser.add_argument('--output', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--model', type=str, default=None, help='ONNXモデルファイルパス')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'directml'], help='デバイス')
    parser.add_argument('--directml', action='store_true', help='DirectMLを使用（Windows GPU）')
    parser.add_argument('--no-video', action='store_true', help='動画を作成しない')

    args = parser.parse_args()

    inference_onnx(
        images_dir=args.images,
        output_dir=args.output,
        model_path=args.model,
        create_video=(not args.no_video),
        device=args.device,
        use_directml=args.directml
    )
