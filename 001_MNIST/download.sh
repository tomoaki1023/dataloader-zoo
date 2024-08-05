#!/bin/bash

# MNISTデータセットのダウンロードと解凍用スクリプト

# ダウンロード先ディレクトリの作成
mkdir -p mnist_data
cd mnist_data

# MNISTデータセットのダウンロード
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# ダウンロードしたファイルの解凍
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "MNISTデータセットのダウンロードと解凍が完了しました。"
