# system python interpreter. used only to create virtual environment
PY = python3
VENV = venv
BIN=$(VENV)/bin
MLFLOW_PATH=mlruns/
HYDRA_MULTI_RUN_PATH=multirun/
HYDRA_OUTPUT_PATH=outputs/

# make it work on windows too
ifeq ($(OS), Windows_NT)
	BIN=$(VENV)/Scripts
	PY=python
endif

$(VENV): requirements.in requirements.dev.in
	$(PY) -m venv $(VENV)
	$(BIN)/pip --no-cache-dir install -U pip
	touch $(VENV)
	touch -a .env

lint:
	$(BIN)/isort ./src
	$(BIN)/black ./src
	$(BIN)/isort ./scripts
	$(BIN)/black ./scripts

clean-env:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

clean-experiments:
	rm -rf $(MLFLOW_PATH)
	rm -rf $(HYDRA_MULTI_RUN_PATH)
	rm -rf $(HYDRA_OUTPUT_PATH)

clean-logs:
	rm -rf $(HYDRA_MULTI_RUN_PATH)
	rm -rf $(HYDRA_OUTPUT_PATH)