"""
PyTorchモデルをONNX形式に変換

RTX 5070 Ti等の新しいGPUでの推論を可能にするため、
PyTorchモデルをONNX形式に変換します。
"""

import os
import torch
import numpy as np

import config
from unet_model import UNetModel


def export_to_onnx(
    model_path=None,
    output_path=None,
    image_size=(512, 512),
    opset_version=14
):
    """
    PyTorchモデルをONNXに変換

    Args:
        model_path: 入力PyTorchモデルパス
        output_path: 出力ONNXモデルパス
        image_size: 入力画像サイズ (height, width)
        opset_version: ONNX opsetバージョン
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if output_path is None:
        output_path = model_path.replace('.pth', '.onnx')

    print(f"\n{'='*60}")
    print(f"  PyTorch → ONNX変換")
    print(f"{'='*60}\n")
    print(f"入力モデル: {model_path}")
    print(f"出力モデル: {output_path}")
    print(f"画像サイズ: {image_size}")
    print(f"ONNX opset: {opset_version}")

    # 入力モデルの存在確認
    if not os.path.exists(model_path):
        print(f"\n❌ エラー: モデルファイルが存在しません: {model_path}")
        raise FileNotFoundError(f"モデルファイルが存在しません: {model_path}")

    # モデルをロード
    print("\n[1/4] モデルを読み込み中...")
    model = UNetModel(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS
    )

    checkpoint = torch.load(model_path, map_location='cpu')

    # state_dictの形式を調整
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # キー名を調整
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.') or key.startswith('decoder.') or key.startswith('segmentation_head.'):
            new_key = 'model.' + key
            new_state_dict[new_key] = value
        elif key.startswith('model.'):
            new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("      ✓ モデル読み込み完了")

    # ダミー入力を作成
    print("\n[2/4] ダミー入力を作成中...")
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
    print("      ✓ ダミー入力作成完了")

    # ONNX変換
    print("\n[3/4] ONNX形式に変換中...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("      ✓ ONNX変換完了")

    # 変換されたモデルのサイズを確認
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      モデルサイズ: {onnx_size:.2f} MB")

    # ONNXモデルの検証
    print("\n[4/4] ONNXモデルを検証中...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("      ✓ ONNXモデルは有効です")
    except ImportError:
        print("      ⚠ onnxパッケージがインストールされていません（検証スキップ）")
    except Exception as e:
        print(f"      ⚠ 検証エラー: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ 変換完了: {output_path}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorchモデルをONNXに変換')
    parser.add_argument('--model', type=str, default=None, help='入力PyTorchモデル（.pth）')
    parser.add_argument('--output', type=str, default=None, help='出力ONNXモデル（.onnx）')
    parser.add_argument('--height', type=int, default=512, help='画像の高さ')
    parser.add_argument('--width', type=int, default=512, help='画像の幅')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opsetバージョン')

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        image_size=(args.height, args.width),
        opset_version=args.opset
    )
