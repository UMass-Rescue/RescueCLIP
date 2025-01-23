SRC = $(wildcard *.py) $(shell find src scripts -type f -name '*.py')

open_clip_smoke_test:
	python scripts/open_clip_smoke_test.py

format:
	black --line-length 110 $(SRC)
	isort --profile black $(SRC)
