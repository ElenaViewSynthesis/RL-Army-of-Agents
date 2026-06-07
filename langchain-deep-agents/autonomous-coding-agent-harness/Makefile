.PHONY: install test test-integration eval run docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

eval:
	python -m evals.retrieval.eval_recall

run:
	python -m agent.main

docker-build:
	docker build -t autonomous-coding-agent-harness .

docker-run:
	docker run --rm --env-file .env autonomous-coding-agent-harness
