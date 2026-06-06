.PHONY: install test eval run run-long docker-build docker-run docker-run-long

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

eval:
	python -m evals.retrieval.eval_recall

run:
	python -m agent.main

run-long:
	python -m agent.main long_horizon

docker-build:
	docker build -t autonomous-coding-agent-harness .

docker-run:
	docker run --rm --env-file .env autonomous-coding-agent-harness

docker-run-long:
	docker run --rm --env-file .env autonomous-coding-agent-harness python -m agent.main long_horizon
