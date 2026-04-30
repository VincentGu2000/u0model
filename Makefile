# Auto-load .env if it exists (for local path configuration)
ifneq (,$(wildcard .env))
    include .env
    export
endif

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	pytest -v --color=yes tests/

.PHONY : format
format :
	isort .
	black .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
