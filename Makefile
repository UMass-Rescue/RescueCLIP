SRC = $(wildcard *.py) $(shell find src scripts -type f -name '*.py')

open_clip_smoke_test:
	python scripts/open_clip_smoke_test.py

profile:
	python -m cProfile -o profile.pstats scripts/open_clip_smoke_test.py
	snakeviz profile.pstats

line_profile:
	kernprof -lv scripts/open_clip_smoke_test.py

format:
	black --line-length 110 $(SRC)
	isort --profile black $(SRC)
