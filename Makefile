install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dataset:
	python -m src.generate_dataset --satellites 12 --requests 500 --horizon-hours 48 --max_windows_per_pair 3 --seed 42

run:
	python -m src.run_experiment --data-dir data/processed --output-dir output --time-limit 30 --workers 8