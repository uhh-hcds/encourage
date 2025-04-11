## Summary of available make targets:
##
## make help         -- Display this message
## make sync-all     -- (Re)install all requisite Python packages including ci/dev
## make format       -- Run code formatter
## make lint         -- Run linter and type checker
## make tests        -- Run tests
## make licenses     -- Summarize licenses used by packages in your venv
##
## This Makefile needs to be run inside a virtual environment.

ifndef VIRTUAL_ENV
ifdef CONDA_PREFIX
$(warning "For better compatibility, consider using a plain Python venv instead of Conda")
VIRTUAL_ENV := $(CONDA_PREFIX)
else
$(error "This Makefile needs to be run inside a virtual environment")
endif
endif

UV_INSTALLED := $(shell command -v uv 2> /dev/null)


uv:
ifeq ($(UV_INSTALLED),)
	echo "uv is not installed, installing it now"
	pip install --upgrade pip
	pip install uv==0.4.28
else
	uv pip install uv==0.4.28
endif
	uv sync --extra ci

help:
	@sed -nE 's/^## ?//p' $(MAKEFILE_LIST)

sync-all:
	uv sync --all-extras

format:
	isort src/encourage src/tests
	ruff format src/g4k src/tests

lint: uv
	ruff check src/encourage src/tests
	# we don't check the typing in the notebooks
	mypy --ignore-missing-imports --incremental src/encourage src/tests

tests: uv
	pytest

licenses:
	pip-licenses --from=mixed --order=license --summary

lock:
	uv lock

.PHONY: help venv format lint tests licenses
