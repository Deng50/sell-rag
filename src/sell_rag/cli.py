from __future__ import annotations

import json
from pathlib import Path

import typer

from sell_rag.app import build_services
from sell_rag.domain import Role
from sell_rag.settings import Settings

app = typer.Typer(help="Sell-RAG local service and administration CLI", no_args_is_help=True)


@app.command()
def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    """Start the local FastAPI service."""
    import uvicorn
    settings = Settings.load()
    uvicorn.run("sell_rag.api.main:create_app", factory=True,
                host=host or settings.host,
                port=port or settings.port, reload=reload)


@app.command()
def ui(url: str = "http://127.0.0.1:8765") -> None:
    """Start the PyQt terminal."""
    from sell_rag.ui import run_ui
    raise typer.Exit(run_ui(url))


@app.command()
def ingest(path: Path, source_id: str | None = None, acl: str = "guest,operator,admin") -> None:
    """Ingest one document and atomically rebuild role indexes."""
    services = build_services()
    roles = [Role(value.strip()) for value in acl.split(",") if value.strip()]
    result = services.ingestion.ingest(path, source_id=source_id, acl=roles, activate=False)
    document = result["document"]
    result["index"] = services.index.build({document["document_id"]: document["version"]})
    document["active"] = True
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def reindex() -> None:
    """Rebuild all role-isolated indexes."""
    typer.echo(json.dumps(build_services().index.build(), ensure_ascii=False, indent=2))


@app.command()
def rollback(document_id: str, version: int) -> None:
    """Activate a ready document version and rebuild indexes."""
    services = build_services()
    services.database.validate_version(document_id, version)
    typer.echo(json.dumps(services.index.build({document_id: version}), ensure_ascii=False, indent=2))


@app.command()
def evaluate(dataset: Path = Path("tests/fixtures/evaluation.jsonl")) -> None:
    """Run the local regression benchmark."""
    typer.echo(json.dumps(build_services().evaluation.run(dataset), ensure_ascii=False, indent=2))


@app.command("migrate-legacy")
def migrate_legacy(path: Path = Path("zhipuai_rag/dataset/chroma_db")) -> None:
    """Import legacy documents.json snapshots without trusting serialized indexes."""
    services = build_services()
    migrated = []
    migration_dir = services.settings.runtime_dir / "legacy-import"
    migration_dir.mkdir(parents=True, exist_ok=True)
    for source in path.rglob("documents.json"):
        rows = json.loads(source.read_text(encoding="utf-8"))
        text = "\n\n".join(str(item.get("content", "")) for item in rows if item.get("content"))
        if not text:
            continue
        target = migration_dir / f"{source.parent.name}.md"
        target.write_text(text, encoding="utf-8")
        migrated.append(services.ingestion.ingest(target, source_id=f"legacy:{source.parent.name}", activate=False))
    changes = {item["document"]["document_id"]: item["document"]["version"] for item in migrated}
    manifest = services.index.build(changes)
    typer.echo(json.dumps({"documents": migrated, "index": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
