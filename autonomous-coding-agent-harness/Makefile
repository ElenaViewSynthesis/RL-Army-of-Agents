.PHONY: install test eval run

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

eval:
	python -m evals.retrieval.eval_recall

run:
	python -m agent.main
