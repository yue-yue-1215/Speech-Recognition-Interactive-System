# 导入所需的模块和库
import json
import pyaudio  # 实时音频采集
import wave  # WAV 文件读写
import threading  # 多线程处理
import numpy as np  # 数组处理
import time  # 时间控制
from queue import Queue  # 队列
import webrtcvad  # 语音活动检测器（Voice Activity Detection）
import os  # 系统路径操作
from transformers import AutoModelForCausalLM, AutoTokenizer  # HuggingFace 模型
from funasr import AutoModel  # ASR 语音识别模型（FunASR）
import pygame  # 用于播放音频
import edge_tts  # 微软 Edge 的 TTS 引擎
import asyncio  # 异步调用支持
import langid  # 语言识别
import websockets  # WebSocket支持
from flask import Flask, jsonify, send_file  # Flask HTTP服务器
import glob  # 文件匹配
import webbrowser  # 自动打开浏览器

# 配置 HuggingFace 国内镜像源，避免网络问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ===== 参数设置 =====
AUDIO_RATE = 16000             # 采样率 16kHz
AUDIO_CHANNELS = 1             # 单声道
CHUNK = 1024                   # 每帧采样点数
VAD_MODE = 3                   # VAD 模式，0-3，越大越灵敏
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR          # 临时保存音频
AUDIO_SAVE_DIR = BASE_DIR      # 保存最终 MP3 和 JSON
NO_SPEECH_THRESHOLD = 1        # 超过该秒数无声即保存语音片段

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# 保存 JSON 的函数
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 初始化 VAD 对象
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# ===== 全局状态变量 =====
recording_active = True            # 录音线程是否运行
last_active_time = time.time()     # 上次有语音活动的时间
segments_to_save = []              # 语音片段缓存（帧 + 时间戳）
saved_intervals = []               # 已保存语音的时间区间
audio_file_count = 0               # 音频编号计数
last_vad_end_time = 0              # 上次保存片段的结束时间

# ===== WebSocket 全局变量 =====
websocket_clients = set()  # 存储所有活跃的WebSocket客户端

# ===== 创建全局事件循环 =====
loop = asyncio.new_event_loop()

# 在单独线程中运行事件循环
def run_event_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

# 启动事件循环线程
loop_thread = threading.Thread(target=run_event_loop, daemon=True)
loop_thread.start()

# ===== WebSocket 发送更新函数 =====
async def send_update(data):
    if websocket_clients:
        for client in list(websocket_clients):
            try:
                await client.send(json.dumps(data))
            except:
                websocket_clients.remove(client)

# ===== WebSocket 服务器处理函数 =====
async def websocket_handler(websocket, path):
    global recording_active
    websocket_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'control':
                recording_active = (data['action'] == 'start')
                await send_update({
                    "type": "status",
                    "message": "正在录音..." if recording_active else "待机中"
                })
    finally:
        websocket_clients.remove(websocket)

# ===== 启动 WebSocket 服务器 =====
async def start_websocket_server():
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    await server.wait_closed()

# ===== Flask 应用初始化 =====
app = Flask(__name__)

# Flask 路由：获取所有日志
@app.route('/logs', methods=['GET'])
def get_logs():
    log_files = glob.glob(os.path.join(AUDIO_SAVE_DIR, "log_*.json"))
    logs = []
    for file in log_files:
        with open(file, 'r', encoding='utf-8') as f:
            logs.append(json.load(f))
    return jsonify(logs)

# Flask 路由：提供音频文件
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(AUDIO_SAVE_DIR, filename))

# Flask 路由：提供前端HTML
@app.route('/')
def serve_frontend():
    return send_file(os.path.join(BASE_DIR, 'index.html'))

# ===== 音频采集线程函数 =====
def record_audio():
    global segments_to_save, last_active_time, last_vad_end_time, audio_file_count

    # 初始化 PyAudio 输入流
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = []  # 每段暂存帧
    print("开始录音...")

    while recording_active:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_buffer.append(data)

        # 每 0.5 秒检查一次是否有语音
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b''.join(audio_buffer)
            if check_vad_activity(raw_audio):
                print("检测到语音")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
                # 通过WebSocket发送状态
                future = asyncio.run_coroutine_threadsafe(
                    send_update({"type": "status", "message": "检测到语音"}),
                    loop
                )
                future.result()  # 等待协程完成
            else:
                print("静音...")
                future = asyncio.run_coroutine_threadsafe(
                    send_update({"type": "status", "message": "静音..."}),
                    loop
                )
                future.result()  # 等待协程完成
            audio_buffer = []  # 清空缓冲区，继续下一段

        # 如果超过阈值时间无语音，保存语音段
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio()
                last_active_time = time.time()

    # 关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

# ===== WebRTC VAD 检测函数 =====
def check_vad_activity(audio_data):
    num = 0
    rate = 0.4  # 40% 的帧为语音则认为整体为语音段
    step = int(AUDIO_RATE * 0.02)  # 每帧 20ms（320字节）

    flag_rate = round(rate * len(audio_data) // step)
    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step and vad.is_speech(chunk, sample_rate=AUDIO_RATE):
            num += 1
    return num > flag_rate

# ===== 保存音频到 .wav 文件并启动模型推理 =====
def save_audio():
    global segments_to_save, saved_intervals, last_vad_end_time, audio_file_count

    if not segments_to_save:
        return

    # 初始化播放引擎（如未初始化）
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # 如果正在播放语音，先打断
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("停止当前播放")

    # 获取当前语音段起止时间
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]

    # 检查与上一次是否有重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("语音段重叠，跳过")
        segments_to_save.clear()
        return

    # 保存为音频文件
    audio_file_count += 1
    audio_path = os.path.join(OUTPUT_DIR, f"audio_{audio_file_count}.wav")
    audio_frames = [seg[0] for seg in segments_to_save]

    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 每个采样点占 2 字节
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频已保存: {audio_path}")

    saved_intervals.append((start_time, end_time))
    segments_to_save.clear()

    # 启动子线程处理语音识别 + LLM + TTS
    threading.Thread(target=run_inference, args=(audio_path,)).start()

# ===== 播放音频文件（MP3） =====
def play_audio(file_path):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print("播放失败：", e)

# ===== TTS 异步函数（微软 edge-tts） =====
async def amain(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# ===== 模型加载：ASR + 大语言模型 =====
asr_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True)  # 小型语音识别模型
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# ===== 语音识别、对话推理、语音合成流程 =====
def run_inference(audio_path):
    global audio_file_count

    # ASR 语音识别
    res = asr_model.generate(input=audio_path, cache={}, language="auto", use_itn=False)
    prompt = res[0]['text'].split('>')[-1].strip() + "，回答简短一些，保持50字以内！"
    print("ASR 识别结果:", prompt)

    # 构造对话上下文
    messages = [
        {"role": "system", "content": "你叫千问，是一个18岁的女大学生，性格活泼开朗，说话俏皮"},
        {"role": "user", "content": prompt},
    ]

    # 构建 prompt 输入
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    # 调用 Qwen 模型生成回答
    output_ids = llm_model.generate(**model_inputs, max_new_tokens=512)
    output_ids = [o[len(i):] for i, o in zip(model_inputs.input_ids, output_ids)]
    output_text = llm_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("回答:", output_text)

    # 判断语言，用于选择合成声音
    lang_code, _ = langid.classify(output_text)
    speakers = {
        "ja": "ja-JP-NanamiNeural",  # 日语
        "fr": "fr-FR-DeniseNeural",  # 法语
        "es": "ca-ES-JoanaNeural",   # 西班牙语
        "zh": "zh-CN-XiaoyiNeural",  # 汉语
        "en": "en-US-AnaNeural"      # 英语
    }
    speaker = speakers.get(lang_code, "zh-CN-XiaoyiNeural")

    # 合成音频输出文件路径
    output_mp3 = os.path.join(AUDIO_SAVE_DIR, f"sft_{audio_file_count}.mp3")

    # 异步调用 TTS 合成音频
    asyncio.run(amain(output_text, speaker, output_mp3))

    # 播放生成的音频
    play_audio(output_mp3)

    # 保存为 JSON 日志
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    json_data = {
        "timestamp": timestamp,
        "audio_file": audio_path,
        "recognized_text": prompt,
        "llm_reply": output_text,
        "language": lang_code,
        "tts_voice": speaker,
        "tts_file": output_mp3
    }
    json_path = os.path.join(AUDIO_SAVE_DIR, f"log_{audio_file_count}.json")
    save_to_json(json_data, json_path)
    print(f"已保存识别记录: {json_path}")

    # 通过WebSocket发送结果
    future = asyncio.run_coroutine_threadsafe(
        send_update({
            "type": "result",
            "asr_result": prompt,
            "llm_reply": output_text,
            "language": lang_code,
            "tts_file": f"/audio/sft_{audio_file_count}.mp3"
        }),
        loop
    )
    future.result()  # 等待协程完成

# ===== 从文本文件合成语音并播放 =====
def text_file_to_speech(txt_path, output_path=None):
    if not os.path.exists(txt_path):
        print("文本文件不存在:", txt_path)
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("文本内容为空")
        return

    # 判断语言选择合适语音
    lang_code, _ = langid.classify(text)
    speakers = {
        "ja": "ja-JP-NanamiNeural",
        "fr": "fr-FR-DeniseNeural",
        "es": "ca-ES-JoanaNeural",
        "zh": "zh-CN-XiaoyiNeural",
        "en": "en-US-AnaNeural"
    }
    speaker = speakers.get(lang_code, "zh-CN-XiaoyiNeural")

    # 合成输出路径
    if output_path is None:
        output_path = os.path.join(AUDIO_SAVE_DIR, f"text2speech_{int(time.time())}.mp3")

    # 合成音频
    asyncio.run(amain(text, speaker, output_path))
    print("语音合成完成:", output_path)

    # 播放
    play_audio(output_path)

# ===== 程序入口：启动音频采集线程和服务器 =====
if __name__ == "__main__":
    try:
        # 启动 WebSocket 服务器
        threading.Thread(target=lambda: asyncio.run(start_websocket_server()), daemon=True).start()
        # 启动 Flask 服务器
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000), daemon=True).start()
        # 自动打开浏览器
        webbrowser.open("http://localhost:8000")
        # 初始化 pygame 和录音线程
        pygame.mixer.init()
        t = threading.Thread(target=record_audio)
        t.start()
        print("按 Ctrl+C 停止...")
        # 主线程保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止...")
        recording_active = False
        t.join()
        loop.call_soon_threadsafe(loop.stop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print("已退出")