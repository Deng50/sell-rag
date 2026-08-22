import os
import time
import numpy as np
import requests
import json
import pyaudio
import base64
# import webrtcvad
# import threading
import urllib.parse
from pydub import AudioSegment
from pydub.playback import play

baidu_API_KEY = os.getenv("BAIDU_API_KEY")
baidu_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")


def _require_baidu_credentials():
    if not baidu_API_KEY or not baidu_SECRET_KEY:
        raise RuntimeError("请先设置 BAIDU_API_KEY 和 BAIDU_SECRET_KEY 环境变量")


def record_audio_non_blocking(filename, silence_threshold=20, silence_duration=5):
    chunk = 1024  # 每次读取的音频块大小
    audio_format = pyaudio.paInt16  # 采样格式
    channels = 1  # 单声道
    rate = 16000  # 采样率

    # 初始化 PyAudio 实例
    p = pyaudio.PyAudio()
    # 打开音频流，准备录音
    stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("开始录音...")

    frames = []  # 存储音频帧
    silent_frames = 0  # 连续静默帧数
    start_time = time.time()

    while True:
        # 读取音频数据块
        data = stream.read(chunk)
        frames.append(data)

        try:
            # 将音频数据转换为 NumPy 数组
            audio_data = np.frombuffer(data, dtype=np.int16)

            # 确保音频数据有效且非空
            if audio_data.size == 0:
                continue

            # 计算音量（RMS），用来判断是否是静默
            volume = np.sqrt(np.mean(audio_data ** 2))
            print(volume)

            # 如果计算出无效值，则将音量设置为0
            if np.isnan(volume) or np.isinf(volume):
                volume = 0  # 设置为0表示静默或无效音量

        except Exception as e:
            # 如果在数据处理过程中发生异常，输出错误信息并跳过当前帧
            print(f"处理音频数据时发生错误: {e}")
            continue

        # 如果音量小于设定的静默阈值，则认为是静默
        if volume < silence_threshold:
            silent_frames += 1  # 增加静默帧数
        else:
            silent_frames = 0  # 重置静默帧数

        # 如果静默持续超过设定的时间，停止录音
        if silent_frames > (rate / chunk * silence_duration):
            print("检测到静默，停止录音。")
            break

    # 停止音频流并关闭 PyAudio 实例
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 合并所有音频帧为一个音频数据
    pcm_data = b''.join(frames)

    # 检查是否存在尾部静默数据，如果有，则去除
    if pcm_data[-chunk:] == b'\x00' * chunk:
        pcm_data = pcm_data[:-chunk]

    # 将录音保存为文件
    with open(filename, 'wb') as f:
        f.write(pcm_data)


# def record_audio_non_blocking(filename, vad=True, timeout=30):
#     chunk = 1024
#     audio_format = pyaudio.paInt16
#     channels = 1
#     rate = 16000
#     frames = []
#
#     def callback(in_data, frame_count, time_info, status):
#         frames.append(in_data)
#         return (in_data, pyaudio.paContinue)
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk, stream_callback=callback)
#
#     print("Start recording...")
#     stream.start_stream()
#
#     # 使用webrtcvad检测声音活动
#     vad_detector = webrtcvad.Vad(1)  # 1 为中等激进性
#
#     start_time = time.time()
#     silence_detected = False
#
#     while time.time() - start_time < timeout:
#         if len(frames) >= 1:
#             frame = frames.pop(0)
#             if len(frame) == chunk:
#                 if not vad_detector.is_speech(frame, rate):
#                     silence_frames_count = 0
#                     while len(frames) > 0 and silence_frames_count < (rate / chunk * 5):
#                         frame = frames.pop(0)
#                         if len(frame) == chunk:
#                             if not vad_detector.is_speech(frame, rate):
#                                 silence_frames_count += 1
#                             else:
#                                 break
#                     if silence_frames_count >= rate / chunk * 5:
#                         silence_detected = True
#                         break
#         time.sleep(0.1)  # 稍微延迟，避免CPU占用过高
#
#     print("Recording finished.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#     pcm_data = b''.join(frames)
#     with open(filename, 'wb') as f:
#         f.write(pcm_data)

def record_audio(filename, duration=7):
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    rate = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("Start recording...")
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    pcm_data = b''.join(frames)
    with open(filename, 'wb') as f:
        f.write(pcm_data)


# def audio_module():
#     audio_file = "../audio.pcm"
#     record_audio(audio_file)
#
#     with open(audio_file, 'rb') as f:
#         audio_data = f.read()
#
#     audio_base64 = base64.b64encode(audio_data).decode('utf-8')
#     token = get_access_token_input()
#
#     payload = {
#         "format": "pcm",
#         "rate": 16000,
#         "channel": 1,
#         "cuid": "WVqnoZsUUTV0synI0zNzJcHYcYVkU21b",
#         "token": token,
#         "len": len(audio_data),
#         "speech": audio_base64
#     }
#
#     headers = {
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }
#
#     response = requests.post("https://vop.baidu.com/server_api", headers=headers, data=json.dumps(payload))
#     result = response.json()
#
#     if result.get("result"):
#         recognized_text = result["result"][0]  # Assuming the first result is the best one
#         print(f"Recognized Text: {recognized_text}")
#         return recognized_text
#     else:
#         print("Error recognizing audio:", result)
#         return None

# audio.py 中的其他代码保持不变

def audio_module():
    audio_file = "../audio.pcm"
    record_audio(audio_file)
    # record_audio_non_blocking(audio_file)

    with open(audio_file, 'rb') as f:
        audio_data = f.read()

    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    token = get_access_token_input()

    payload = {
        "format": "pcm",
        "rate": 16000,
        "channel": 1,
        "cuid": "WVqnoZsUUTV0synI0zNzJcHYcYVkU21b",
        "token": token,
        "len": len(audio_data),
        "speech": audio_base64
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.post("https://vop.baidu.com/server_api", headers=headers, data=json.dumps(payload))
    result = response.json()


    if result.get("result"):
        recognized_text = result["result"][0]  # Assuming the first result is the best one
        print(f"Recognized Text: {recognized_text}")
        return recognized_text
    else:
        print("Error recognizing audio:", result)
        return None

def get_access_token_input():
    _require_baidu_credentials()
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": baidu_API_KEY, "client_secret": baidu_SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


# def audio_output(text):
#     max_length = 500  # 假设最大长度为500个字符
#     for i in range(0, len(text), max_length):
#         segment = text[i:i + max_length]
#         url = "https://tsn.baidu.com/text2audio"
#         encoded_text = urllib.parse.quote(segment)
#         payload = f'tex={encoded_text}&tok={get_access_token_output()}&cuid=tFgcFMAtGEKg7hPuaESS9emu2tR0XY9a&ctp=1&lan=zh&spd=5&pit=5&vol=5&per=1&aue=6'
#
#         headers = {
#             'Content-Type': 'application/x-www-form-urlencoded',
#             'Accept': '*/*'
#         }
#
#         try:
#             response = requests.post(url, headers=headers, data=payload)
#
#             if response.status_code == 200:
#                 content_type = response.headers.get('Content-Type')
#                 if "audio" in content_type:
#                     audio_file_path = "../output.wav"
#                     with open(audio_file_path, "wb") as f:
#                         f.write(response.content)
#                     audio = AudioSegment.from_file(audio_file_path)
#                     play(audio)
#                     print("语音合成完成，已播放音频")
#                 else:
#                     print("合成失败，返回的内容不是音频格式，Content-Type:", content_type)
#                     print("错误信息：", response.json())
#             else:
#                 print(f"请求失败，状态码：{response.status_code}，内容：{response.text}")
#
#         except Exception as e:
#             print(f"发生错误：{e}")

def audio_output(text):
    max_length = 500
    for i in range(0, len(text), max_length):
        segment = text[i:i + max_length]
        url = "https://tsn.baidu.com/text2audio"
        encoded_text = urllib.parse.quote(segment)
        payload = f'tex={encoded_text}&tok={get_access_token_output()}&cuid=tFgcFMAtGEKg7hPuaESS9emu2tR0XY9a&ctp=1&lan=zh&spd=7&pit=5&vol=5&per=4119&aue=6'

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': '*/*'
        }

        try:
            response = requests.post(url, headers=headers, data=payload)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                if "audio" in content_type:
                    audio_file_path = "../output.wav"
                    with open(audio_file_path, "wb") as f:
                        f.write(response.content)
                    audio = AudioSegment.from_file(audio_file_path)
                    play(audio)
                    print("语音合成完成，已播放音频")

                    # 播放完毕后开始录音
                    # threading.Thread(target=record_audio_non_blocking, args=("../response.pcm",)).start()
                    # threading.Thread(target=record_audio, args=("../response.pcm",)).start()
                else:
                    print("合成失败，返回的内容不是音频格式，Content-Type:", content_type)
                    print("错误信息：", response.json())
            else:
                print(f"请求失败，状态码：{response.status_code}，内容：{response.text}")

        except Exception as e:
            print(f"发生错误：{e}")

# def audio_output_auto_record(text):
#     max_length = 500
#     for i in range(0, len(text), max_length):
#         segment = text[i:i + max_length]
#         url = "https://tsn.baidu.com/text2audio"
#         encoded_text = urllib.parse.quote(segment)
#         payload = f'tex={encoded_text}&tok={get_access_token_output()}&cuid=tFgcFMAtGEKg7hPuaESS9emu2tR0XY9a&ctp=1&lan=zh&spd=5&pit=5&vol=5&per=1&aue=6'
#
#         headers = {
#             'Content-Type': 'application/x-www-form-urlencoded',
#             'Accept': '*/*'
#         }
#
#         try:
#             response = requests.post(url, headers=headers, data=payload)
#
#             if response.status_code == 200:
#                 content_type = response.headers.get('Content-Type')
#                 if "audio" in content_type:
#                     audio_file_path = "../output.wav"
#                     with open(audio_file_path, "wb") as f:
#                         f.write(response.content)
#                     audio = AudioSegment.from_file(audio_file_path)
#                     play(audio)
#                     print("语音合成完成，已播放音频")
#
#                     # 播放完毕后开始录音
#                     # threading.Thread(target=record_audio_non_blocking, args=("../response.pcm",)).start()
#                     # threading.Thread(target=audio_module).start()
#                     audio_module()
#                 else:
#                     print("合成失败，返回的内容不是音频格式，Content-Type:", content_type)
#                     print("错误信息：", response.json())
#             else:
#                 print(f"请求失败，状态码：{response.status_code}，内容：{response.text}")
#
#         except Exception as e:
#             print(f"发生错误：{e}")

def get_access_token_output():
    _require_baidu_credentials()
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": baidu_API_KEY, "client_secret": baidu_SECRET_KEY}
    response = requests.post(url, params=params)
    return str(response.json().get("access_token"))
