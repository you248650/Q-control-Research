# DRL for Quantum Control

## ディレクトリ構造
```
.
├── .devcontainer                          	# devcontainer
│   └── devcontainer.json		           # devcontainerでの設定(VSCodeの設定/拡張機能)
├── Docker		                        # Dockerでの環境構築
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── requirements_dependence.txt
├── Double_harmonic_oscillator_model		# DRL学習と結果の保存
│   ├── custom_gym_env
│   │   ├── __init__.py
│   │   └── envs
│   │       ├── __init__.py
│   │       ├── double_harmonic_oscillator.py	  # 演算子等の定義
│   │       ├── double_quantum_environment.py	  # DRL環境
│   │       └── smesolve.py                       # SMEの自作solver
│   └── main.py
├── Plot_Result                                 # 結果の描画
│   ├── density_matrix_plot.py                    # Wigner関数
│   ├── fidelity_plot.py                          # Fidelity曲線
│   └── reward_plot.py                            # Reward曲線
├── run.sh
└── tensorboard.sh
```
<br/>

## シミュレーションの実行
### 環境構築
#### DevContainerを用いる場合
1. [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (VSCode拡張機能) をインストール

#### DevContainerを用いない場合
```sh
sudo docker compose --file Docker/docker-compose.yml up --detach --build
``` 
> [!NOTE]
> ```Notes```と```Results```のVolumeが生成 

### 実行
1. Containerにアクセス
```sh
sudo docker container exec -it RL4Quantum
```
2. 実行
```sh
sh run.sh
```

### 結果の保存先
- ```Notes```: TensorBoard用の記録 (```tensorboard.sh```で実行可能)
- ```Results```: PIDごと、エピソードごと、ステップ単位で密度行列, Action, Reward, Terminalを記録
<br/>

## Serverの環境構築
### 必要なソフトウェア
 - [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) / [CUDA](https://developer.nvidia.com/cuda-toolkit)
 - [Docker](https://docs.docker.com/engine/install/ubuntu/) / [Docker Compose](https://docs.docker.com/engine/install/ubuntu/)
 - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
 - [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/)

### インストールの確認
```sh
nvidia-smi  # NVIDIA Driver & CUDA 確認
docker --version  # Docker 確認
docker-compose --version  # Docker Compose 確認
```

### 必要に応じて再インストール
#### 1. 既存のNVIDIA Driver / CUDA / cuDNNの削除
```sh
sudo apt-get update && \
sudo apt-get purge -y nvidia-* && \
sudo apt-get purge -y coda-* && \
sudo apt-get autoremove && \
sudo apt-get autoclean && \
sudo rm -rf /usr/local/cuda*
```

#### 2. 既存のDocker / Docker Composeの削除
```sh
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done && \
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras && \
sudo rm -rf /var/lib/docker && \
sudo rm -rf /var/lib/containerd
```

#### 3. NVIDIA Driverのinstall
1. GPU対応のNVIDIA Driverの確認
```sh
ubuntu-drivers devices
```
(出力例)
```
== /sys/devices/pci0000:00/0000:00:1e.0 ==
modalias : pci:v000010DEd00001EB8sv000010DEsd000012A2bc03sc02i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-460-server - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-460 - distro non-free recommended
driver   : nvidia-driver-450 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

2. 1.で確認したNVIDIA Driverのversionのうち、”recommended”のversion(上記の例では、nvidia-driver-460)をinstall
```sh
sudo apt-get -y update && \
sudo apt-get install -y [recommended version] && \
sudo reboot
```

#### 4. Docker / Docker Composeのinstallのinstall
```sh
udo apt-get update && \
sudo apt-get install -y ca-certificates curl && \
sudo install -m 0755 -d /etc/apt/keyrings && \
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
sudo chmod a+r /etc/apt/keyrings/docker.asc && \
echo \ "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \ $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \ sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
sudo apt-get update && \
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
sudo docker run hello-world
```

#### 5. NVIDIA Container Toolkitのinstall
1. NVIDIA Container Toolkitのinstall
```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \ && \
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \ sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \ sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
sudo apt-get update && \
sudo apt-get install -y nvidia-container-toolkit && \
sudo nvidia-ctk runtime configure --runtime=docker && \
sudo systemctl restart docker && \
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

2. 設定ファイルの編集
```diff :/etc/nvidia-container-runtime/config.toml
- no-cgroups = true
+ no-cgroups = false
```

#### 6. cuDNNのinstall
```sh
sudo apt-get update && \
sudo apt-get -y install libcudnn9-samples && \
cd $HOME/cudnn_samples_v9/mnistCUDNN && \
make clean && \
make ./mnistCUDNN
```