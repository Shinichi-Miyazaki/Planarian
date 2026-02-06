"""
U-Netモデルの定義

segmentation_models_pytorchを使用して、
ResNet34バックボーン + ImageNet事前学習済み重みで転移学習
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetModel(nn.Module):
    """
    U-Netセグメンテーションモデル

    ResNet34エンコーダー + ImageNet事前学習済み重みで初期化
    """

    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=3, out_channels=1):
        """
        Args:
            encoder_name: エンコーダーのアーキテクチャ（resnet34推奨）
            encoder_weights: 事前学習済み重み（'imagenet'推奨）
            in_channels: 入力チャンネル数（RGB=3）
            out_channels: 出力チャンネル数（バイナリセグメンテーション=1）
        """
        super(UNetModel, self).__init__()

        # segmentation_models_pytorchのU-Netを使用
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None  # Sigmoidは損失関数内で適用
        )

    def forward(self, x):
        """順伝播"""
        return self.model(x)


class DiceBCELoss(nn.Module):
    """
    Dice Loss + Binary Cross Entropy Loss の組み合わせ

    セグメンテーションタスクで高精度を実現するための損失関数
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            dice_weight: Dice Lossの重み
            bce_weight: BCE Lossの重み
            smooth: ゼロ除算回避用の微小値
        """
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: モデルの出力（logits）
            target: 正解マスク（0 or 1）

        Returns:
            loss: 総合損失
        """
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Sigmoidを適用
        pred_sigmoid = torch.sigmoid(pred)

        # Dice Loss
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                    (pred_flat.sum() + target_flat.sum() + self.smooth)

        # 重み付き総合損失
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return total_loss


def build_model(encoder_name='resnet34', encoder_weights='imagenet',
                in_channels=3, out_channels=1, device='cuda'):
    """
    モデルをビルドしてデバイスに転送

    Args:
        encoder_name: エンコーダー名
        encoder_weights: 事前学習済み重み
        in_channels: 入力チャンネル数
        out_channels: 出力チャンネル数
        device: デバイス（'cuda' or 'cpu'）

    Returns:
        model: 構築されたモデル
    """
    model = UNetModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        out_channels=out_channels
    )

    model = model.to(device)

    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"モデルを構築しました:")
    print(f"  - エンコーダー: {encoder_name}")
    print(f"  - 事前学習済み重み: {encoder_weights}")
    print(f"  - 総パラメータ数: {total_params:,}")
    print(f"  - 学習可能パラメータ数: {trainable_params:,}")
    print(f"  - デバイス: {device}")

    return model


def load_model(model_path, encoder_name='resnet34', in_channels=3,
               out_channels=1, device='cuda'):
    """
    保存済みモデルを読み込み

    Args:
        model_path: モデルファイルのパス
        encoder_name: エンコーダー名
        in_channels: 入力チャンネル数
        out_channels: 出力チャンネル数
        device: デバイス

    Returns:
        model: 読み込まれたモデル
    """
    # モデルを構築（重みは読み込まない）
    model = UNetModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        out_channels=out_channels
    )

    # 保存された重みを読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # state_dictの形式を確認して読み込み
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"モデルを読み込みました（Epoch {checkpoint.get('epoch', 'N/A')}）")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  - Val Dice: {checkpoint.get('val_dice', 'N/A')}")
    else:
        state_dict = checkpoint
        print(f"モデルを読み込みました")

    # キー名の調整（互換性対応）
    # 保存時: encoder.*, decoder.*, segmentation_head.*
    # 期待: model.encoder.*, model.decoder.*, model.segmentation_head.*
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.') or key.startswith('decoder.') or key.startswith('segmentation_head.'):
            # model. プレフィックスを追加
            new_key = 'model.' + key
            new_state_dict[new_key] = value
        elif key.startswith('model.'):
            # 既にmodel.プレフィックスがある場合はそのまま
            new_state_dict[key] = value
        else:
            # その他のキーもそのまま
            new_state_dict[key] = value

    # strict=Falseで読み込み（num_batches_trackedなどの不一致を許容）
    model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model
