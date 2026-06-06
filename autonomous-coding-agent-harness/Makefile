.PHONY: install test test-integration eval eval-e2e run run-long docker-build docker-run docker-run-long

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

eval:
	python -m evals.retrieval.eval_recall

eval-e2e:
	python -m evals.e2e.eval_fixture_task

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
