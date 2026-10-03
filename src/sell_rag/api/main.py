import asyncio
import hmac
import json
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sell_rag.app import Services, build_services
from sell_rag.domain import Product, QueryRequest, Role
from sell_rag.multimodal import SpeechNormalizer
from sell_rag.settings import Settings


class FeedbackBody(BaseModel):
    request_id: str | None = None
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2000)


class EvaluationBody(BaseModel):
    dataset: str = "tests/fixtures/evaluation.jsonl"


class SpeechSynthesisBody(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    services = services or build_services(settings)
    app = FastAPI(title="Sell-RAG API", version="1.0.0")
    app.state.services = services

    def role_dependency(
        authorization: Annotated[str | None, Header()] = None,
        x_role: Annotated[str, Header()] = "guest",
    ) -> Role:
        try:
            role = Role(x_role.lower())
        except ValueError as exc:
            raise HTTPException(400, "无效角色") from exc
        if role == Role.GUEST:
            return role
        scheme, _, token = (authorization or "").partition(" ")
        expected = services.settings.admin_token if role == Role.ADMIN else services.settings.operator_token
        if (scheme.lower() != "bearer" or not expected
                or not hmac.compare_digest(token.strip().encode("utf-8"), expected.encode("utf-8"))):
            raise HTTPException(401, "管理令牌无效或未配置")
        return role

    def require(minimum: Role):
        order = {Role.GUEST: 0, Role.OPERATOR: 1, Role.ADMIN: 2}

        def dependency(role: Annotated[Role, Depends(role_dependency)]) -> Role:
            if order[role] < order[minimum]:
                raise HTTPException(403, "权限不足")
            return role
        return dependency

    def rebuild_job(job_id: str, changes: dict[str, int | None] | None = None) -> None:
        try:
            services.database.update_job(job_id, "running", 0.2, "正在构建隔离索引")
            manifest = services.index.build(changes)
            services.database.update_job(job_id, "complete", 1.0, json.dumps(manifest, ensure_ascii=False))
        except Exception as exc:
            services.database.update_job(job_id, "failed", 1.0, str(exc))
    def ingest_job(job_id: str, path: Path, source_id: str, owner: str, acl: list[Role]) -> None:
        try:
            services.database.update_job(job_id, "running", 0.1, "正在解析文档")
            result = services.ingestion.ingest(path, source_id=source_id, owner=owner, acl=acl, activate=False)
            services.database.update_job(job_id, "running", 0.7, "正在更新索引")
            document = result["document"]
            manifest = services.index.build({document["document_id"]: document["version"]})
            document["active"] = True
            detail = {"ingestion": result, "index": manifest}
            services.database.update_job(job_id, "complete", 1.0, json.dumps(detail, ensure_ascii=False))
        except Exception as exc:
            services.database.update_job(job_id, "failed", 1.0, str(exc))
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "mode": "fake" if services.settings.fake_providers else "live",
            "generator": services.answer.generator.name, "vision": services.vision.name,
            "speech": services.speech.name, "embedding": services.index.embedding.name,
            "reranker": services.index.reranker.name, "degraded": services.index.degraded,
        }

    @app.get("/ready")
    def ready() -> dict:
        try:
            services.database.list_documents()
            return {"ready": True, "active_index": services.database.active_index()}
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/v1/query")
    def query_answer(body: QueryRequest, role: Annotated[Role, Depends(role_dependency)]):
        return services.answer.answer(body.model_copy(update={"role": role}))

    @app.post("/v1/query/stream")
    def query_stream(body: QueryRequest, role: Annotated[Role, Depends(role_dependency)]):
        answer = services.answer.answer(body.model_copy(update={"role": role}))

        async def events():
            yield f"event: meta\ndata: {json.dumps({'request_id': answer.request_id, 'route': answer.route.value})}\n\n"
            for start in range(0, len(answer.answer), 12):
                piece = answer.answer[start:start + 12]
                yield f"event: token\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            yield f"event: citations\ndata: {json.dumps([item.model_dump(mode='json') for item in answer.citations], ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {answer.model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/documents", status_code=202)
    async def upload_document(
        background: BackgroundTasks,
        role: Annotated[Role, Depends(require(Role.OPERATOR))],
        file: UploadFile = File(...),
        source_id: str | None = None,
        acl: str = "guest,operator,admin",
    ) -> dict:
        safe_name = (file.filename or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1]
        if safe_name in {"", ".", ".."}:
            raise HTTPException(400, "文件名无效")
        try:
            roles = [Role(item.strip()) for item in acl.split(",") if item.strip()]
        except ValueError as exc:
            raise HTTPException(400, "ACL 包含无效角色") from exc
        if not roles:
            raise HTTPException(400, "ACL 不能为空")
        incoming = services.settings.runtime_dir / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        maximum = services.settings.max_file_mb * 1024 * 1024
        content = await file.read(maximum + 1)
        if len(content) > maximum:
            raise HTTPException(413, f"文件超过 {services.settings.max_file_mb} MB 限制")
        job_id = services.database.create_job("upload")
        target = incoming / job_id / safe_name
        target.parent.mkdir()
        target.write_bytes(content)
        background.add_task(ingest_job, job_id, target, source_id or safe_name, role.value, roles)
        return {"job_id": job_id, "filename": safe_name}

    @app.get("/v1/documents")
    def documents(_: Annotated[Role, Depends(require(Role.OPERATOR))]):
        return services.database.list_documents()

    @app.post("/v1/documents/{document_id}/versions/{version}/activate", status_code=202)
    def activate(document_id: str, version: int, background: BackgroundTasks,
                 _: Annotated[Role, Depends(require(Role.ADMIN))]):
        try:
            services.database.validate_version(document_id, version)
        except KeyError as exc:
            raise HTTPException(404, "文档版本不存在") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        job_id = services.database.create_job("reindex", f"activate {document_id}:{version}")
        background.add_task(rebuild_job, job_id, {document_id: version})
        return {"job_id": job_id}

    @app.delete("/v1/documents/{document_id}", status_code=202)
    def delete_document(document_id: str, background: BackgroundTasks,
                        _: Annotated[Role, Depends(require(Role.ADMIN))]):
        if not any(row["document_id"] == document_id for row in services.database.list_documents()):
            raise HTTPException(404, "文档不存在")
        job_id = services.database.create_job("reindex", f"delete {document_id}")
        background.add_task(rebuild_job, job_id, {document_id: None})
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str, _: Annotated[Role, Depends(require(Role.OPERATOR))]):
        result = services.database.get_job(job_id)
        if not result:
            raise HTTPException(404, "任务不存在")
        return result

    @app.get("/v1/security/findings")
    def security_findings(_: Annotated[Role, Depends(require(Role.ADMIN))]):
        return services.database.prompt_injection_findings()

    @app.get("/v1/products")
    def products(
        role: Annotated[Role, Depends(role_dependency)],
        text: str | None = None, category: str | None = None,
        min_price: Decimal | None = Query(None, ge=0, allow_inf_nan=False),
        max_price: Decimal | None = Query(None, ge=0, allow_inf_nan=False),
        in_stock: bool = False, limit: int = Query(20, ge=1, le=200),
    ):
        if min_price is not None and max_price is not None and min_price > max_price:
            raise HTTPException(400, "最低价格不能高于最高价格")
        return services.database.query_products(
            text=text, category=category,
            min_price=Decimal(str(min_price)) if min_price is not None else None,
            max_price=Decimal(str(max_price)) if max_price is not None else None,
            in_stock=in_stock, limit=limit, role=role,
        )

    @app.post("/v1/products")
    def upsert_product(product: Product, _: Annotated[Role, Depends(require(Role.ADMIN))]):
        services.database.upsert_product(product)
        return {"sku": product.sku, "status": "saved"}

    @app.post("/v1/speech/transcribe")
    async def transcribe(role: Annotated[Role, Depends(role_dependency)], audio: UploadFile = File(...),
                         sample_rate: Literal[8000, 16000] = 16000):
        maximum = sample_rate * 2 * 60
        content = await audio.read(maximum + 1)
        if len(content) > maximum:
            raise HTTPException(413, "音频超过 60 秒 PCM 上限")
        if not content:
            raise HTTPException(400, "音频不能为空")
        transcript = await asyncio.to_thread(services.speech.transcribe, content, sample_rate)
        hotwords = [product.name for product in services.database.query_products(role=role, limit=10000)]
        return SpeechNormalizer(hotwords, services.settings.asr_confirm_threshold).normalize(transcript)

    @app.post("/v1/speech/synthesize")
    def synthesize(body: SpeechSynthesisBody, role: Annotated[Role, Depends(role_dependency)]):
        del role
        return StreamingResponse(iter([services.speech.synthesize(body.text)]), media_type="audio/wav")

    @app.post("/v1/feedback")
    def feedback(body: FeedbackBody, role: Annotated[Role, Depends(role_dependency)]):
        del role
        return {"feedback_id": services.database.add_feedback(body.request_id, body.rating, body.comment)}

    @app.post("/v1/evaluations")
    def evaluate(body: EvaluationBody, _: Annotated[Role, Depends(require(Role.OPERATOR))]):
        path = Path(body.dataset).resolve()
        allowed = [Path("tests/fixtures").resolve(),
                   (services.settings.runtime_dir / "evaluations").resolve()]
        if not any(path.is_relative_to(directory) for directory in allowed):
            raise HTTPException(403, "评测文件必须位于 tests/fixtures 或 runtime/evaluations 目录")
        if not path.is_file():
            raise HTTPException(404, "评测数据不存在")
        if path.stat().st_size > services.settings.max_file_mb * 1024 * 1024:
            raise HTTPException(413, "评测数据过大")
        try:
            return services.evaluation.run(path)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"评测数据格式错误: {exc}") from exc

    return app
