# 语音识别交互系统 README

## 项目介绍

本项目是一个基于实时音频采集、语音活动检测（VAD）、自动语音识别（ASR）、大语言模型（LLM）以及文本转语音（TTS）的前后端结合的一体式交互语音系统。系统通过麦克风实时采集音频，检测语音活动，使用 ASR 模型将音频转为文本，调用大语言模型生成回答，并通过 TTS 将回答合成为语音播放，同时通过 Web 界面展示实时结果和历史日志。以下是项目的详细说明、依赖库、文件结构以及运行方式。

## 项目功能

- **实时音频采集**：通过 PyAudio 从麦克风采集音频。
- **语音活动检测（VAD）**：使用 WebRTC VAD 检测语音段，分离有效语音和静音。
- **语音识别（ASR）**：基于 FunASR 的 SenseVoiceSmall 模型，将音频转为文本。
- **对话生成**：使用 Qwen2.5-1.5B-Instruct 大语言模型生成自然语言回答。
- **语音合成（TTS）**：通过 Microsoft Edge TTS 将文本合成为多语言语音。
- **Web 界面**：基于 Flask 和 WebSocket，提供实时状态、结果展示和历史日志查询。
- **日志管理**：保存每次交互的音频、识别文本、模型回答和 TTS 文件信息。

## 环境要求

- **操作系统**：Windows Anaconda Prompt 中运行。
- **Python 版本**：虚拟 conda 环境 python==3.10。
- **硬件要求**：麦克风设备（用于音频采集）。
- **网络要求**：需要联网下载预训练模型和 TTS 语音数据。

## 目录展示

project_directory/
├── main.py              # 主程序，包含音频采集、VAD、ASR、LLM、TTS 和服务器逻辑
├── tools                # 提供基础环境
├── index.html           # 前端 HTML 页面，展示实时状态、结果和历史日志
├── audio_X.wav          # 临时保存的音频片段（运行时生成，X 为序号）
├── sft_X.mp3            # TTS 生成的语音文件（运行时生成，X 为序号）
├── log_X.json           # 每次交互的日志文件（运行时生成，X 为序号）
├── text2speech_X.mp3    # 从文本文件合成的语音（运行时生成，X 为时间戳）
除 main.py 、tools和 index.html 后面几个文件均随项目运行自动生成。

## 运行步骤

1. 创建虚拟环境。
    ```bash
    conda create -n chatAudio python=3.10
    conda activate chatAudio
    ```
2. 安装 pytorch 系列。
    ```bash
    pip install "D:\pytorch_wheels\torch-2.3.1+cu118-cp310-cp310-win_amd64.whl"
    pip install "D:\pytorch_wheels\torchvision-0.18.1+cu118-cp310-cp310-win_amd64.whl"
    pip install "D:\pytorch_wheels\torchaudio-2.3.1+cu118-cp310-cp310-win_amd64.whl"
    ```
3. 安装项目运行所需的 Python 库。
    ```bash
    pip install pyaudio webrtcvad numpy transformers funasr pygame edge-tts langid websockets flask
    ```
4. 完成模型下载。
    - SenseVoiceSmall 模型下载。
        ```bash
        https://www.modelscope.cn/models/iic/SenseVoiceSmall/files [^21^]
        ```
        同时设置代码 model_dir = "iic/SenseVoiceSmall"。
    - QWen 模型下载。
        ```bash
        https://www.modelscope.cn/models/ [^21^]
        ```
        同时设置代码 model_name = "Qwen/Qwen2.5-1.5B-Instruct"。
5. 项目运行。
    ```bash
    conda activate chatAudio
    python d:/1/lab_final/main.py（这是我的地址）
    ```
6. 项目效果。
    - 实时语音转文字，启动 WebSocket 服务器（端口 8765）用于实时通信，启动 Flask 服务器（端口 8000）提供 Web 界面和 API。
    - 自动打开浏览器访问 http://localhost:8000 可以看到日志（显示当前语音识别结果、模型回答、检测语言和 TTS 音频文件（可点击播放））。
    - 此外在左侧目录生成每次记录 json 和每次录音（包括用户问的 audio_x.wav 和 ai 答的 sft_x.mp3）。
    - 文字转语音同理。
    - TTS 语言支持：系统支持中文、英文、日文、法文、西班牙文等多种语言的 TTS，语言由 langid 自动检测，用户可以说其中的任何一种语言并得到其中任何一种语言的反馈。
    - 按 Ctrl+C 终止程序关闭音频流和事件循环。
    - 详见视频展示。