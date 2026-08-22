from chromadb import Embeddings
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import torch


class XiaobuEmbedding(Embeddings):
    def __init__(self):
        # self.tokenizer_name = "./lier007/xiaobu-embedding-v2"
        self.tokenizer_name = "./RAG/xiaobu-embedding-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.tokenizer_name)

    def embed_documents(self, texts):
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.embed_single_text, texts))
        return embeddings

    def embed_single_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding[0].numpy()

    def embed_query(self, query):
        # 为查询生成嵌入
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding[0].numpy()
