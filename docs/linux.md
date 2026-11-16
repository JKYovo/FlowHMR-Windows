# Linux 源码运行（NVIDIA）

已在 Ubuntu 22.04 x86_64、NVIDIA RTX 3060 12 GB 上验证视频动捕、接触校正及 NPZ/FBX 导出。
该脚本使用 Python 3.10 和 PyTorch 2.5.1+cu118；需要兼容 CUDA 11.8 的 NVIDIA 驱动。
其他 Linux 发行版和显卡尚未验证，Windows 便携包的运行环境管理不适用于此安装方式。

## 安装

需要 Git、`uv`、C++17 编译器及常用 OpenCV 系统库。Ubuntu 可先安装：

```bash
sudo apt install git build-essential libgl1 libglib2.0-0
```

按 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，再把仓库克隆到有足够空间的位置：

```bash
git clone https://github.com/xiaoerofficial021-collab/FlowHMR-Windows.git
cd FlowHMR-Windows
./setup-linux.sh
./download-models.sh --all
```

安装脚本只初始化推理所需的 YOLOX、SAM 3D Body 和 VGGT 子模块，应用现有 SAM 推理适配，
并编译 Kimodo/RGBMocap 接触校正组件。FBX SDK 使用上游指定的 Linux CPython 3.10 wheel，固定 SHA-256。
`requirements-linux.lock.txt` 记录已验证的依赖版本，`requirements-linux.txt` 记录依赖来源及额外约束。
YOLOX 按上游使用 `--no-deps` 安装；`pip check` 可能列出未安装的训练/ONNX 导出依赖，
它们不用于网页动捕，运行环境检查请使用 `./check-install.sh`。

`--all` 下载所有公开推理模型（含躯干优化及 HaMeR），沿用仓库的固定版本与 SHA-256 校验。
不带 `--all` 仅下载基础推理模型；可通过 `--groups flowhmr sam` 等选择组别。
SMPL-H/MANO 仍需按 [模型说明](../models/README.md) 取得并通过网页导入，脚本不提供这些许可模型的镜像。

Python、`.venv`、模型、缓存及任务结果均保存在仓库目录中，便于安装到独立磁盘。
首次安装需要下载数 GB 依赖和模型；移动目录后应重新运行安装脚本重建含绝对路径的环境。

## 启动和停止

```bash
./open-flowhmr.sh       # 后台启动并打开浏览器；重复运行复用已有服务
./status-web.sh
./stop-web.sh
```

默认地址为 <http://127.0.0.1:8080>，仅监听本机。前台运行可用 `./start-web.sh`，按 Ctrl+C 停止。
`FLOWHMR_PORT=8081 ./open-flowhmr.sh` 可指定端口。服务使用进程 ID 和启动时间验证所属进程，避免误停其他程序。
该入口使用 CUDA:0；默认固定机位、自动 SAM 批量，手动批量初始值为 4。

启动脚本隔离外部 `PYTHONPATH`/`PYTHONHOME`，启用 UTF-8，并将模型与临时缓存放到项目目录。
分块注意力沿用 Windows 启动设置：`query_batched`，每块 256 个查询，降低长序列生成显存峰值。
这与 SAM 批次数量独立，不截断动作序列。可先停止服务，再用
`FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE=128 ./open-flowhmr.sh` 进一步减小生成阶段的显存占用。

## 视频及验证

- FFmpeg 转码和裁剪日志按 UTF-8 解码，避免中文路径在 ASCII 环境中导致失败。
- 奇数尺寸视频在右侧/底部补边到偶数尺寸，避免 H.264 失败后退回浏览器不支持的 MPEG-4。
- 旧任务使用单独的 H.264 兼容预览，不更改其原有推理视频、相机参数或 SAM 缓存。
- 显式深度模型校验重新读取文件，避免 NTFS 上同尺寸快速覆盖复用过时的散列缓存。

实机验证覆盖 30、1060 和 4480 帧，最后一例使用 SAM 批量 8、手部每 4 帧采样、固定机位、4 步生成。
4480 帧任务复用已有 SAM 特征后成功完成；生成阶段按秒采样到的进程显存峰值约 7.0 GiB，
NPZ 有限值检查和 FBX 重新导入通过。1157×643 的 WebM 旧任务生成 1158×644 H.264 预览后，同步播放通过。
这些是特定素材和设置下的实测，不代表所有视频都能在同样显存内运行。

任务目录为 `output/web_jobs/`，后台服务日志为 `logs/linux-web.log`。
`./check-install.sh --environment-only` 可只检查环境，`./check-install.sh` 同时检查模型。
