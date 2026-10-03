import logging

from sell_rag import cli
from sell_rag.observability import configure_logging


def test_serve_loads_models_only_in_api_factory(services, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli.Settings, "load", lambda: services.settings)
    monkeypatch.setattr(cli, "build_services", lambda: (_ for _ in ()).throw(AssertionError("duplicate model load")))
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.serve()
    assert calls[0][1]["factory"] is True
    assert calls[0][1]["port"] == services.settings.port


def test_reconfiguring_logs_closes_owned_handler_without_touching_other_handlers(tmp_path) -> None:
    configure_logging(tmp_path / "first")
    root = logging.getLogger()
    first = next(handler for handler in root.handlers if getattr(handler, "_sell_rag_handler", False))
    configure_logging(tmp_path / "first")
    assert sum(getattr(handler, "_sell_rag_handler", False) for handler in root.handlers) == 1
    configure_logging(tmp_path / "second")
    assert first.stream is None
    assert first not in root.handlers
