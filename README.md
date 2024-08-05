# Data Loader Zoo

Data Loader Zooは、深層学習プロジェクトのためのフレームワーク非依存のデータローダーコレクションです。様々なデータセットに対応し、異なる深層学習フレームワークで利用可能なPythonベースの実装を提供します。

## 概要

深層学習プロジェクトでは、データセットを適切な形式でニューラルネットワークに入力する必要があります。しかし、以下の課題が存在します：

- データセットごとにデータローダーを作成する手間
- 深層学習フレームワークごとに異なるデータ形式への対応

Data Loader Zooは、これらの課題を解決し、研究者や開発者の作業効率を向上させることを目的としています。

## 特徴

- フレームワーク非依存：主要な深層学習フレームワークで利用可能
- 豊富なデータセット対応：一般的なデータセットから特殊なものまで幅広くサポート
- 簡単に使える：統一されたインターフェースで簡単に利用可能
- カスタマイズ可能：必要に応じて拡張や修正が可能

## 使い方

```python
from data_loader_zoo import MNISTLoader

# MNISTデータセットのローダーを初期化
loader = MNISTLoader(batch_size=32, shuffle=True)

# データの取得
for batch_x, batch_y in loader:
    # ここで学習処理を行う
    ...