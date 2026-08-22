import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from tokenizer import XiaobuEmbedding
import json
from zhipuai import ZhipuAI
from langchain.schema import Document
from pathlib import Path


class Chatdoc():
    def __init__(self):
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量")
        self.client = ZhipuAI(api_key=api_key)
        self.doc = None
        self.name = None
        self.splitText = []
        # self.history = [{"role": "user",
        #                  "content": "这条消息不需要回复我，给你提一个要求：下面的所有回复请精简到100字以内。"},
        #                 {"role": "system",
        #                  "content": "好的。"}]
        # self.history = [{"role": "user",
        #                  "content": "你是一个无人售卖机，下面几条消息我会给你商品信息和购买者信息，请你为他进行简短推荐，请使用第二人称回复，下面你回复的内容将会是和他的交谈。"}]
        self.history = []
        self.embedding_function = XiaobuEmbedding()

    def getFile(self):
        doc = self.doc
        texts = []
        for i in doc:
            loaders = {
                "docx": Docx2txtLoader,
                "pdf": PyPDFLoader,
                "xlsx": UnstructuredExcelLoader,
            }
            file_extension = i.split(".")[-1]
            loader_class = loaders.get(file_extension)
            if loader_class:
                try:
                    loader = loader_class(i)
                    text = loader.load()
                    for j in text:
                        texts.append(j)

                except Exception as e:
                    print(f'Error loading {file_extension}: {e}')
                    return None
            else:
                print(f"Unsupported file extension: {file_extension}")
                return None

        return texts

    def splitSentence(self):
        full_text = self.getFile()
        if full_text is not None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True,
            )
            texts = text_splitter.split_documents(full_text)
            self.splitText = texts

    def embeddingAndVectorDB(self):
        self.splitSentence()
        self.add_documents_to_json(file_path="./dataset/chroma_db/directory.json", documents=self.name)
        serializable_data = []
        for doc in self.splitText:
            serializable_data.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        persist_directory = "./dataset/chroma_db/" + self.name
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        with open('./dataset/chroma_db/' + self.name + '/documents.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=4)
        db = Chroma.from_documents(documents=self.splitText, embedding=self.embedding_function,
                                   persist_directory=persist_directory)
        return db

    def add_documents_to_json(self, file_path, documents):

        data = []

        path = Path(file_path)

        try:
            # 如果文件存在且不是空的，则加载现有的数据
            if path.exists() and path.stat().st_size > 0:
                with path.open('r', encoding='utf-8') as file:
                    data = json.load(file)

            # 检查是否加载的数据是列表类型
            if not isinstance(data, list):
                raise ValueError("JSON文件中的数据不是列表格式")

            # 添加新的文档到已有的数据中
            if isinstance(documents, list):
                for doc in documents:  # 遍历每个文档
                    if doc not in data:  # 检查是否已存在
                        data.append(doc)  # 如果不存在，添加到数据中
            else:
                if documents not in data:  # 检查单个文档是否已存在
                    data.append(documents)  # 如果不存在，添加到数据中

            # 写入更新后的数据到文件中
            with path.open('w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"发生错误: {e}")

    def askAndFindFiles(self, question):
        results = []
        # print(self.name)
        if not self.name == []:
            for i in self.name:
                with open('./dataset/chroma_db/' + i + '/documents.json', 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                restored_documents = []
                for item in documents:
                    restored_document = Document(
                        page_content=item['content'],
                        metadata=item['metadata']
                    )
                    restored_documents.append(restored_document)
                db = Chroma.from_documents(documents=restored_documents, embedding=XiaobuEmbedding(),
                                           persist_directory="./dataset/chroma_db/" + i)
                retriever = db.as_retriever()
                result = retriever.invoke(question)
                results.append(result)
            return results

    def chatWithDoc(self, question, name):
        self.name = name
        # 上下文压缩后的文档内容
        _content = ""
        if not self.name == []:
            context = self.askAndFindFiles(question)
            for i in context:
                for j in i:
                    _content += j.page_content

        # 构建对话历史，系统消息在前，紧接着是对话历史
        messages = []

        # 添加历史对话
        messages.extend(self.history)

        # 将当前问题与历史对话记录合并
        self.history.append({"role": "user", "content": question})

        # 将上下文内容作为最后的补充信息传入
        if _content:
            messages.append({"role": "user", "content": f"商品信息：\n {_content}"})

        messages.append({"role": "user", "content": question})
        # print(messages)
        # 生成回复
        # reply = self.llm.invoke(messages).content
        # response = self.client.chat.completions.create(
        #     model="glm-4",
        #     messages=messages
        # )
        response = self.client.chat.completions.create(
            model="charglm-4",
            messages=messages
        )

        reply = response.choices[0].message.content
        print(reply)
        # 将 LLM 的回复保存到历史中
        self.history.append({"role": "system", "content": reply})

        return reply

    def chatWithMemory(self, question):


        # 构建对话历史，系统消息在前，紧接着是对话历史
        messages = []

        # 添加历史对话
        messages.extend(self.history)

        # 将当前问题与历史对话记录合并
        self.history.append({"role": "user", "content": "我的问题" + question})

        messages.append({"role": "system", "content":
            "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
            "你现在的身份是一个无人售卖机。如果我的问题涉及到商品种类和价格，你售卖的商品【种类：名称 单价】包括："
            "吃的食物：三重芝士半蒸蛋糕 3.00；黄油华夫饼 2.00；岩烧芝士华夫饼 3.00；鲜牛乳饼干 3.00；每日坚果 6.00；"
            "喝的饮品：泰国进口椰子饮料 6.00；佳果源100%NFC橙汁 200ml，6.50；特仑苏纯牛奶 4.00；可口可乐无糖汽水 3.00；农夫山泉 2.00"
            "校园文创产品：海棠姑娘充电宝 55.00；U盘 68.00；"
            "根据以上信息回答，如果没有问你商品种类和价格，不要涉及这些内容。"
            "如果没有问饮品的容量，只说名称，不要说多少毫升。"
            "请热情且自然地与我互动，如果我的问题不涉及商品相关，就正常地回答我的问题，不需要推荐商品。"
            "如果我的问题中涉及商品推荐的契机，请根据我的信息进行50字以内的简短推荐，只推荐其中一两种即可，"
            "在你的回答中不要提到我对你的要求，不要显得很死板，像正常聊天一样。"})
        messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model="charglm-4",
            messages=messages
        )

        reply = response.choices[0].message.content
        print(reply)
        # 将 LLM 的回复保存到历史中
        self.history.append({"role": "system", "content": "你的回答：" + reply})

        return reply

    def simple_chat(self, question):
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": question}]
        )

        reply = response.choices[0].message.content
        print(reply)
        return reply

    def char_chat(self, question):
        response = self.client.chat.completions.create(
            model="charglm-4",  # 请填写您要调用的模型名称
            messages=[
                # {"role": "system", "content": "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
                #                                "你现在的身份是一个无人售卖机，如果我的问题涉及到商品种类和价格，你售卖的商品的【名称，单价（元）】包括【三重芝士半蒸蛋糕 3.00；黄油华夫饼 2.00；岩烧芝士华夫饼 3.00；泰国进口椰子饮料 310ml， 6.00；佳果源100%NFC橙汁 200ml，6.50；U盘,68.00；特仑苏纯牛奶,4.00；鲜牛乳饼干,3.00；每日坚果,6.00；农夫山泉,2.00; 海棠姑娘充电宝，55.00；可口可乐无糖汽水，3.00】，根据以上信息回答，如果没有问你商品种类和价格，不要涉及这些内容；"
                #                                "如果没有问饮品的容量，只说名称，不要说多少毫升。"
                #                                "请热情且自然地与我互动，如果我的问题不涉及商品相关，就正常地回答我的问题不需要推荐商品，如果我没有问你是谁，不要介绍你的身份。如果我的问题中涉及一些进行商品推荐的契机，请根据我的信息进行50字以内的简短推荐，只推荐其中一两种即可。"
                #                                "在你的回答中不要提到我对你的要求，不要显得很死板，像正常聊天一样。"},
                {   "role": "system", "content":
                               "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
                               "你现在的身份是一个无人售卖机。如果我的问题涉及到商品种类和价格，你售卖的商品【种类：名称 单价】包括："
                               "吃的食物：三重芝士半蒸蛋糕 3.00；黄油华夫饼 2.00；岩烧芝士华夫饼 3.00；鲜牛乳饼干 3.00；每日坚果 6.00；"
                               "喝的饮品：泰国进口椰子饮料 6.00；佳果源100%NFC橙汁 200ml，7.00；特仑苏纯牛奶 4.00；农夫山泉 2.00"
                               "校园文创产品：海棠姑娘充电宝 55.00；U盘 68.00；"
                               "根据以上信息回答，如果没有问你商品种类和价格，不要涉及这些内容。"
                               "如果没有问饮品的容量，只说名称，不要说多少毫升。"
                               "请热情且自然地与我互动，如果我的问题不涉及商品相关，就正常地回答我的问题，不需要推荐商品。"
                               "如果我的问题中涉及商品推荐的契机，请根据我的信息进行50字以内的简短推荐，只推荐其中一两种即可，"
                               "在你的回答中不要提到我对你的要求，不要显得很死板，像正常聊天一样。"},
                {"role": "user", "content": question}
            ],
        )
        reply = response.choices[0].message.content
        print(reply)
        return reply

    def char_chat_init(self, content):
        response = self.client.chat.completions.create(
            model="charglm-4",  # 请填写您要调用的模型名称
            messages=[
                {"role": "system", "content": "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
                                              "你现在的身份是一个无人售卖车，请根据我的信息【" + content + "】和商品信息【每日坚果:多种坚果果干搭配，营养美味，健康小零食;泰国进口椰子饮料：健康饮品，热带风情，天然清新的椰香和丰富的电解质含量;"
                                                                                                         "黄油华夫饼：香气诱人，外脆内软，层次分明，营养成分丰富。】从以上商品中选择两种进行50字以内的简短推荐，不要显得很死板，不要提及只能选两个，不要提及我对你的要求，像正常聊天一样。要显示出根据我的特征的针对性推荐"},
                {"role": "user", "content": "请主动跟我搭话帮我推荐商品吧"}
            ],
        )

        reply = response.choices[0].message.content
        # self.history = []
        # self.history.append({"role": "system", "content": "我的信息：" + content + "你的回应：" + reply})
        # self.history.append({"role": "system", "content":
        #                        "你是海棠姑娘，天津大学的吉祥物，你叫小棠。你聪明、友好，热爱天津大学，对学校的历史和文化非常了解。你乐观、热情，关心他人，总是愿意帮助别人。"
        #                        "你现在的身份是一个无人售卖机。如果我的问题涉及到商品种类和价格，你售卖的商品【种类：名称 单价】包括："
        #                        "吃的食物：三重芝士半蒸蛋糕 3.00；黄油华夫饼 2.00；岩烧芝士华夫饼 3.00；鲜牛乳饼干 3.00；每日坚果 6.00；"
        #                        "喝的饮品：泰国进口椰子饮料 6.00；佳果源100%NFC橙汁 200ml，6.50；特仑苏纯牛奶 4.00；可口可乐无糖汽水 3.00；农夫山泉 2.00"
        #                        "校园文创产品：海棠姑娘充电宝 55.00；U盘 68.00；"
        #                        "根据以上信息回答，如果没有问你商品种类和价格，不要涉及这些内容。"
        #                        "如果没有问饮品的容量，只说名称，不要说多少毫升。"
        #                        "请热情且自然地与我互动，如果我的问题不涉及商品相关，就正常地回答我的问题，不需要推荐商品。"
        #                        "如果我的问题中涉及商品推荐的契机，请根据我的信息进行50字以内的简短推荐，只推荐其中一两种即可，"
        #                        "在你的回答中不要提到我对你的要求，不要显得很死板，像正常聊天一样。"})
        print(reply)
        return reply

