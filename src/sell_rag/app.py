from __future__ import annotations

from dataclasses import dataclass

from sell_rag.evaluation import EvaluationRunner
from sell_rag.generation import AnswerService, FakeGenerator, FakeVision, ZhipuGenerator, ZhipuVision
from sell_rag.indexing import IndexManager
from sell_rag.ingestion import IngestionService
from sell_rag.multimodal import BaiduSpeech, FakeSpeech
from sell_rag.observability import configure_logging
from sell_rag.settings import Settings
from sell_rag.storage import Database


@dataclass
class Services:
    settings: Settings
    database: Database
    index: IndexManager
    ingestion: IngestionService
    answer: AnswerService
    speech: object
    vision: object
    evaluation: EvaluationRunner


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings.load()
    configure_logging(settings.runtime_dir)
    database = Database(settings.database)
    index = IndexManager(settings, database)
    if settings.fake_providers or not settings.zhipuai_api_key:
        generator, vision = FakeGenerator(), FakeVision()
        if not settings.fake_providers and not settings.zhipuai_api_key:
            index.degraded.append("未配置 ZHIPUAI_API_KEY，使用 Fake 生成与视觉")
    else:
        generator, vision = ZhipuGenerator(settings.zhipuai_api_key), ZhipuVision(settings.zhipuai_api_key)
    if settings.fake_providers or not (settings.baidu_api_key and settings.baidu_secret_key):
        speech = FakeSpeech()
        if not settings.fake_providers:
            index.degraded.append("未配置百度语音密钥，使用 Fake 语音")
    else:
        speech = BaiduSpeech(settings.baidu_api_key, settings.baidu_secret_key,
                             settings.asr_confirm_threshold)
    ingestion = IngestionService(settings, database, vision.describe)
    answer = AnswerService(settings, database, index, generator)
    evaluation = EvaluationRunner(answer, index, database)
    return Services(settings, database, index, ingestion, answer, speech, vision, evaluation)
