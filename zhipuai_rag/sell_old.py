import os
import time
import cv2
import torch
import base64
from zhipuai import ZhipuAI
from audio import (audio_module, audio_output)
# , audio_output_auto_record)
import threading

from rag import Chatdoc


class yolo():
    def __init__(self):
        self.chat_doc = Chatdoc()

    # def detect_and_capture(self, model_path='weights/yolov5s.pt', screenshot_path='person_detected.png',
    #                        confidence_threshold=0.8, area_threshold=20000, delay=0.5):
    #     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    #
    #     # 打开摄像头
    #     cap = cv2.VideoCapture(0)
    #
    #     # 截图标志位
    #     screenshot_taken = False
    #
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("无法读取摄像头数据")
    #             break
    #
    #         # 模型推理
    #         results = model(frame)
    #         content = None
    #         # 解析检测结果
    #         detections = results.pandas().xyxy[0]
    #         for _, row in detections.iterrows():
    #             if (
    #                     row['name'] == 'person' and
    #                     row['confidence'] > confidence_threshold and
    #                     (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']) > area_threshold
    #                     and not screenshot_taken
    #             ):
    #                 # print("检测到人，等待1秒后截图...")
    #                 time.sleep(delay)  # 延时
    #
    #                 # 重新读取摄像头的最新画面
    #                 ret, new_frame = cap.read()
    #                 if not ret:
    #                     print("无法读取延时后的摄像头画面")
    #                     break
    #
    #                 # 保存最新的帧
    #                 cv2.imwrite(screenshot_path, new_frame)
    #                 # print(f"截图已保存至 {screenshot_path}")
    #                 screenshot_taken = True
    #                 content = self.image_llm(screenshot_path)
    #
    #         # 可视化检测结果
    #         annotated_frame = results.render()[0]
    #         cv2.imshow('YOLOv5 Realtime Detection', annotated_frame)
    #
    #         # # 按 'q' 键退出
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         if content is not None:
    #             break
    #
    #     # 释放资源
    #     cap.release()
    #     cv2.destroyAllWindows()

    def detect_and_capture(self, model_path='weights/yolov5s.pt', screenshot_path='person_detected.png',
                           confidence_threshold=0.8, area_threshold=20000, delay=0.5):
        # 加载自定义训练的YOLOv5模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        # 打开摄像头
        cap = cv2.VideoCapture(0)

        # 截图标志位
        screenshot_taken = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头数据")
                break

            # 模型推理
            results = model(frame)
            content = None

            # 解析检测结果
            detections = results.pandas().xyxy[0]
            max_area = 0
            max_area_person = None

            for _, row in detections.iterrows():
                # 只关心'person'类且满足设定的置信度和面积阈值
                area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
                if row['name'] == 'person' and row['confidence'] > confidence_threshold and area > area_threshold:
                    # 选择面积最大的'person'外接框
                    if area > max_area:
                        max_area = area
                        max_area_person = row

            # 如果检测到目标且是面积最大的人的外接框
            if max_area_person is not None and not screenshot_taken:
                # print("检测到最大面积的人，等待1秒后截图...")
                time.sleep(delay)  # 延时

                # 重新读取摄像头的最新画面
                ret, new_frame = cap.read()
                if not ret:
                    print("无法读取延时后的摄像头画面")
                    break

                # 保存最新的帧
                cv2.imwrite(screenshot_path, new_frame)
                # print(f"截图已保存至 {screenshot_path}")
                screenshot_taken = True
                content = self.image_llm(screenshot_path)

            # 可视化检测结果
            # annotated_frame = results.render()[0]
            # cv2.imshow('YOLOv5 Realtime Detection', annotated_frame)

            # # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # 如果已经成功保存截图，跳出循环
            if content is not None:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

    def image_llm(self, image_path):
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量")
        client = ZhipuAI(api_key=api_key)
        img_path = image_path
        with open(img_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="glm-4v-plus",  # 需要调用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base
                            }
                        },
                        {
                            "type": "text",
                            "text": "请详细描述图片中主要人物的相关特征，包括性别、年龄、服饰、心情等"
                        }
                    ]
                }
            ]
        )
        content = response.choices[0].message.content
        print(response.choices[0].message.content)
        self.get_response(
            "我的信息：" + content + "你是一个无人售卖机，请根据商品信息和我的信息进行100字以内的简短推荐，不要显得很死板，像正常聊天一样。",
            ["sell"])
        return content
        # print(response)

    def get_response(self, prompt, name):
        if prompt:
            ai_message = self.chat_doc.chatWithDoc(prompt, name)

            # threading.Thread(target=audio_output, args=(ai_message,)).start()
            audio_output(ai_message)
    def start_conversation(self):
        while True:
            # 获取用户的语音输入并转换为文本
            user_input = audio_module()
            if user_input is None:
                ai_message = self.chat_doc.simple_chat(
                    "你是一个无人售卖机，请热情跟我道别，不要显得很死板，像正常聊天一样。")
                audio_output(ai_message)
                print("未能识别到语音输入，结束对话。")
                self.detect_and_capture()
                break

            # 检查是否结束对话
            elif user_input.lower() in ["再见。", "结束。", "停止。", ""]:
                ai_message = self.chat_doc.simple_chat(
                    "我的问题：" + user_input + "你是一个无人售卖机，请热情跟我道别，不要显得很死板，像正常聊天一样。")
                audio_output(ai_message)
                print("对话结束。")
                self.detect_and_capture()
                break
            else:
                # 将用户输入的文本转换为语音并播放
                ai_message = self.chat_doc.chatWithDoc(
                    "我的问题：" + user_input + "你是一个无人售卖机，请根据商品信息和我的信息进行100字以内的简短推荐，不要显得很死板，像正常聊天一样。",
                    ["sell"])
                audio_output(ai_message)

# 调用函数
if __name__ == "__main__":
    yolo().detect_and_capture()
    while True:
        try:
            yolo().start_conversation()
        except KeyboardInterrupt:
            print("程序退出")
