PYTHON_DIRS := src web_app
PYTHON_ROOT_SCRIPTS := $(wildcard *.py)

.PHONY: venv install freeze-requirements pipeline web flake8 ruff black lint format

venv:
	python3 -m venv .venv

install:
	pip install -r requirements.txt

freeze-requirements:
	pip freeze > requirements-frozen.txt

pipeline:
	python -m src.run_pipeline

web:
	streamlit run web_app/streamlit_app.py

flake8:
	python -m flake8 $(PYTHON_DIRS) $(PYTHON_ROOT_SCRIPTS)

ruff:
	python -m ruff check $(PYTHON_DIRS) $(PYTHON_ROOT_SCRIPTS)

black:
	python -m black --check $(PYTHON_DIRS) $(PYTHON_ROOT_SCRIPTS)

lint: flake8 ruff black

format:
	python -m black $(PYTHON_DIRS) $(PYTHON_ROOT_SCRIPTS)
