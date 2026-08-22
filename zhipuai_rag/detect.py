import os
import time
import cv2
import torch
import base64
from zhipuai import ZhipuAI


def detect_and_capture(model_path='weights/yolov5s.pt', screenshot_path='liyifeng.png',
                       confidence_threshold=0.9, area_threshold=20000, delay=1):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

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

        # 解析检测结果
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            if (
                    row['name'] == 'person' and
                    row['confidence'] > confidence_threshold and
                    (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']) > area_threshold
                    and not screenshot_taken
            ):
                print("检测到人，等待1秒后截图...")
                time.sleep(delay)  # 延时

                # 重新读取摄像头的最新画面
                ret, new_frame = cap.read()
                if not ret:
                    print("无法读取延时后的摄像头画面")
                    break

                # 保存最新的帧
                cv2.imwrite(screenshot_path, new_frame)
                print(f"截图已保存至 {screenshot_path}")
                screenshot_taken = True
                image_llm(screenshot_path)

        # 可视化检测结果
        annotated_frame = results.render()[0]
        cv2.imshow('YOLOv5 Realtime Detection', annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def image_llm(image_path):
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
    # print(response.choices[0].message.content)
    return response.choices[0].message.content
    # print(response)


# 调用函数
if __name__ == "__main__":
    detect_and_capture()
