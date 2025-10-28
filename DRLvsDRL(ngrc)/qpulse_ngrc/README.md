# qpulse-ngrc

NG-RC をナビ（前ふるい）として使い、Policy Gradient で制御パルス（I/Q）を最適化する研究用パッケージです。
- Qiskit + Aer + IBM Runtime（ノイズモデル）
- TensorFlow / TFP（方策勾配）
- scikit-learn（NG-RC / Ridge）

## インストール

```bash
pipx run build   # もしくは: python -m build
pip install dist/qpulse_ngrc-0.1.0-py3-none-any.whl
```

開発モードなら：

```bash
pip install -e .
```

## 使い方（CLI）

```bash
qpulse-run --help
qpulse-run --cycles 500 --batch-size 32 --use-nav 1
```

- IBM Cloud 資格情報はコードに**埋め込まず**、環境変数を利用してください。

```bash
export IBM_QUANTUM_TOKEN="***"
export IBM_QUANTUM_INSTANCE="crn:v1:bluemix:public:quantum-computing:us-east:..."
```

資格情報が取得できない場合は、**ローカル Aer（空ノイズ）**に自動フォールバックします。

## データキャッシュ

学習用データセット（NAV_dataset_timeseries.pkl）は、`NGRC_DATA_DIR` が指定されていればそこ、
無ければ OS の一時ディレクトリ配下に作られます。

```bash
export NGRC_DATA_DIR="$HOME/.cache/qpulse_ngrc"
```

## 免責

研究用途の参考実装です。互換性は環境に依存します。必要に応じて依存バージョンを調整してください。
