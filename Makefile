
# Python interpreter inside the virtual environment (path differs between Windows and Linux/macOS).
ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

# Defaults; override on the command line, e.g. `make run TIME_LIMIT=60`.
TIME_LIMIT ?= 30
WORKERS ?= 8
SEED ?= 42

# These targets are command names, not files.
.PHONY: help venv install dataset datasets preview run run-no-hint benchmark benchmark-xlarge benchmark-no-hint test lint format typecheck check all clean

help:                ## Show this list of commands
	@echo "make venv              create the .venv virtual environment"
	@echo "make install           install requirements into .venv"
	@echo "make dataset           generate data/processed/*.parquet (SEED=$(SEED))"
	@echo "make datasets          generate the 4 benchmark scenarios -> data/benchmark/{tiny,small,medium,large}/"
	@echo "make preview           print the first/last 200 rows of each table"
	@echo "make run               run CP-SAT + 2 greedy on data/processed -> output/"
	@echo "make run-no-hint       same, CP-SAT without greedy warm start"
	@echo "make benchmark         tiny/small/medium/large scenarios -> output/benchmark_results.csv"
	@echo "make benchmark-no-hint large scenario without warm start (shows why the hint matters)"
	@echo "make benchmark-xlarge  30 satellites x 5000 requests, 120 s time limit (slow)"
	@echo "make test              run the pytest suite"
	@echo "make lint / format     ruff check / ruff format"
	@echo "make typecheck         mypy on src/"
	@echo "make check             test + lint"
	@echo "make all               dataset + run + test"
	@echo "make clean             delete generated data, outputs and caches"
	@echo "Options: TIME_LIMIT=$(TIME_LIMIT) WORKERS=$(WORKERS) SEED=$(SEED)"

# ---- Setup -------------------------------------------------------------------

venv:
	python -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ---- Data --------------------------------------------------------------------

dataset:
	$(PYTHON) -m src.generate_dataset --satellites 12 --requests 500 --horizon_hours 48 --max_windows_per_pair 3 --seed $(SEED)

# Only generate the benchmark datasets (no solving). Sizes are defined in SCENARIOS in src/benchmark.py.
# Preview one with: python -m src.preview_dataset --data-dir data/benchmark/large
datasets:
	$(PYTHON) -m src.benchmark --generate-only

preview:
	$(PYTHON) -m src.preview_dataset --data-dir data/processed --rows 200

# ---- Experiments -------------------------------------------------------------

run:
	$(PYTHON) -m src.run_experiment --data-dir data/processed --output-dir output --time-limit $(TIME_LIMIT) --workers $(WORKERS)

run-no-hint:
	$(PYTHON) -m src.run_experiment --data-dir data/processed --output-dir output --time-limit $(TIME_LIMIT) --workers $(WORKERS) --no-hint

benchmark:
	$(PYTHON) -m src.benchmark --time-limit $(TIME_LIMIT) --workers $(WORKERS)

benchmark-no-hint:
	$(PYTHON) -m src.benchmark --scenarios large --no-hint --time-limit $(TIME_LIMIT) --workers $(WORKERS) --output output/benchmark_nohint.csv

benchmark-xlarge:
	$(PYTHON) -m src.benchmark --scenarios xlarge --time-limit 120 --workers $(WORKERS) --output output/benchmark_xlarge.csv

# ---- Quality -----------------------------------------------------------------

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

check: test lint

# ---- Shortcuts ---------------------------------------------------------------

# Full pipeline from scratch: generate data, run the 3 algorithms, run the tests.
all: dataset run test

# Cross-platform delete (Python instead of rm/del, which differ between shells).
clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['output', 'data/processed', 'data/benchmark', '.pytest_cache', '.ruff_cache', '.mypy_cache']]"
