.PHONY: install install-tts install-vlm test lint typecheck check doctor dry-run

install:
	uv sync --extra dev

install-tts:
	scripts/bootstrap.sh --tts

install-vlm:
	scripts/bootstrap.sh --vlm

test:
	uv run pytest -q -m "not slow"

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

check: test lint typecheck

doctor:
	uv run python -m xrbench doctor

dry-run:
	uv run python -m xrbench all --dry-run
