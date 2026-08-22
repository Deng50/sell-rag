import os
import time
import cv2
import torch
import base64
from zhipuai import ZhipuAI
from audio import (audio_module, audio_output)  # 导入音频模块，用于语音输入和输出
from rag import Chatdoc

class yolo():
    """
    一个用于检测图像中人类并进行语音交互的类。该类集成了YOLOv5目标检测模型，
    用于检测摄像头中的人类并捕捉截图。同时，它也集成了ZhipuAI API，进行图像分析。
    """

    def __init__(self):
        # 初始化时创建一个 Chatdoc 实例，用于后续的对话管理。
        self.chat_doc = Chatdoc()

    def detect_and_capture(self, model_path='weights/yolov5s.pt', screenshot_path='person_detected.png',
                           confidence_threshold=0.8, area_threshold=20000, delay=0.5):
        """
        使用YOLOv5模型实时检测摄像头图像中的人类，并根据设定的条件捕获截图。
        - model_path: YOLO模型文件路径
        - screenshot_path: 截图保存路径
        - confidence_threshold: 置信度阈值，过滤低于该值的检测
        - area_threshold: 面积阈值，过滤过小的目标
        - delay: 延迟时间，用于在检测到人类后稍等片刻再截图
        """
        # 加载自定义训练的YOLOv5模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        # 打开摄像头
        cap = cv2.VideoCapture(0)

        # 截图标志位，防止重复截图
        screenshot_taken = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头数据")
                break

            # 使用YOLO模型进行推理
            results = model(frame)
            content = None

            # 解析YOLO模型的检测结果
            detections = results.pandas().xyxy[0]
            max_area = 0
            max_area_person = None

            # 遍历所有检测到的目标，寻找符合条件的“person”
            for _, row in detections.iterrows():
                # 计算目标的面积，过滤掉过小或置信度较低的目标
                area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
                if row['name'] == 'person' and row['confidence'] > confidence_threshold and area > area_threshold:
                    # 选择面积最大的目标（人类）
                    if area > max_area:
                        max_area = area
                        max_area_person = row

            # 如果检测到符合条件的人类，且未截图过，进行截图
            if max_area_person is not None and not screenshot_taken:
                time.sleep(delay)  # 延迟，等待新的帧出现

                # 重新读取摄像头图像
                ret, new_frame = cap.read()
                if not ret:
                    print("无法读取延时后的摄像头画面")
                    break

                # 保存当前图像为截图
                cv2.imwrite(screenshot_path, new_frame)
                screenshot_taken = True
                content = self.image_llm(screenshot_path)  # 调用图像分析函数

            #可视化检测结果（注释掉了显示窗口的部分，可以根据需要打开）
            annotated_frame = results.render()[0]
            cv2.imshow('YOLOv5 Realtime Detection', annotated_frame)


            # 如果已经成功保存截图，跳出循环
            if content is not None:
                break
            # # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 释放摄像头资源并关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()

    def image_llm(self, image_path):
        """
        调用 ZhipuAI API 对图像进行分析，并返回图像中的主要人物特征。
        - image_path: 图像文件路径
        """
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量")
        client = ZhipuAI(api_key=api_key)
        img_path = image_path
        # 读取图像并将其转换为Base64格式
        with open(img_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')

        # 调用ZhipuAI进行图像分析
        response = client.chat.completions.create(
            model="glm-4v-plus",  # 使用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base  # 图像的Base64字符串
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
        # 获取并打印ZhipuAI的响应内容
        content = response.choices[0].message.content
        print(content)

        # 调用 get_response 方法继续对话
        # self.get_response(
        #     "我的信息：【" + content + "】你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
        #                             "你现在的身份是一个无人售卖车，请根据我的信息和商品信息【每日坚果:多种坚果果干搭配，营养美味，健康小零食;泰国进口椰子饮料：健康饮品，热带风情，天然清新的椰香和丰富的电解质含量;"
        #                             "黄油华夫饼：香气诱人，外脆内软，层次分明，营养成分丰富。】从以上商品中选择两种进行50字以内的简短推荐，不要显得很死板，不要提及只能选两个，不要提及我对你的要求，像正常聊天一样。",
        #     ["sell"])
        self.get_response_init(content)
        return content

    def get_response(self, prompt, name):
        """
        根据输入的提示，获取AI的回复，并将其通过语音输出。
        - prompt: 用户的输入文本
        - name: 用于控制对话场景的参数（例如：['sell']表示售卖机的对话）
        """
        if prompt:
            # 获取AI回复
            ai_message = self.chat_doc.chatWithDoc(prompt, name)
            # 语音输出AI的回答
            audio_output(ai_message)

    def get_response_init(self, content):
        """
        根据输入的提示，获取AI的回复，并将其通过语音输出。
        - prompt: 用户的输入文本
        - name: 用于控制对话场景的参数（例如：['sell']表示售卖机的对话）
        """

        ai_message = self.chat_doc.char_chat_init(content)
        # 语音输出AI的回答
        audio_output(ai_message)

    def start_conversation(self):
        """
        启动语音对话流程。此方法会持续获取用户的语音输入，生成AI回复并输出语音。
        """
        while True:
            # 获取用户的语音输入并转换为文本
            user_input = audio_module()
            if user_input is None:
                # 如果没有识别到语音，结束对话并调用检测功能
                ai_message = self.chat_doc.simple_chat(
                    "你叫小棠，请热情且很简洁的跟我道别，不要很死板，像正常聊天一样。")
                audio_output(ai_message)
                print("未能识别到语音输入，结束对话。")
                self.detect_and_capture()  # 调用检测和截图功能
                break

            # 检查是否结束对话（通过语音输入判断）
            elif user_input.lower() in ["再见。", "结束。", "停止。", ""] or "再见" in user_input.lower():
                ai_message = self.chat_doc.simple_chat(
                    "我的道别语：" + user_input + "你叫小棠，请热情且简洁地跟我道别。")
                audio_output(ai_message)
                print("对话结束。")
                self.detect_and_capture()  # 调用检测和截图功能
                break
            # else:
            #     # 如果输入的是有效问题，生成AI回复并通过语音输出
            #     ai_message = self.chat_doc.chatWithDoc(
            #         "我的问题：" + user_input + "你是一个无人售卖机，请根据商品信息和我的信息进行100字以内的简短推荐，不要显得很死板，像正常聊天一样。",
            #         ["sell"])
            #     audio_output(ai_message)
            else:
                # 如果输入的是有效问题，生成AI回复并通过语音输出
                # ai_message = self.chat_doc.simple_chat(
                #     "我的问题：" + user_input + "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
                #                                "你现在的身份是一个无人售卖机，如果我的问题涉及到商品种类和价格，你售卖的商品的【名称，单价（元）】包括【 岩烧芝士华夫饼 3.00；泰国进口椰子饮料 310ml， 6.00；佳果源100%NFC橙汁 200ml，6.50；U盘,68.00；特仑苏纯牛奶,4.00；鲜牛乳饼干,3.00；每日坚果,6.00；农夫山泉,2.00; 海棠姑娘充电宝，88.00】，根据以上信息回答，如果没有问你商品种类和价格，不要涉及这些内容；"
                #                                "如果没有问饮品的容量，只说名称，不要说多少毫升。"
                #                                "请热情且自然地与我互动，如果我的问题不涉及商品相关，就正常地回答我的问题不需要推荐商品，如果我没有问你是谁，不要介绍你的身份。如果我的问题中涉及一些进行商品推荐的契机，请根据我的信息进行50字以内的简短推荐，只推荐其中一两种即可。"
                #                                "在你的回答中不要提到我对你的要求，不要显得很死板，像正常聊天一样。")
                # ai_message = self.chat_doc.chatWithMemory(user_input)
                ai_message = self.chat_doc.char_chat(user_input)
                audio_output(ai_message)

# 主程序入口
if __name__ == "__main__":
    # 实例化 yolo 类并开始进行检测和截图
    yolo().detect_and_capture()
    while True:
        try:
            # 启动语音对话
            yolo().start_conversation()
        except KeyboardInterrupt:
            # 处理退出程序的情况
            print("程序退出")
