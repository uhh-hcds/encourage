## Summary of available make targets:
##
## make help         -- Display this message
## make -B venv      -- (Re)install all requisite Python packages
## make format       -- Run code formatter
## make lint         -- Run linter and type checker
## make tests        -- Run tests
## make requirements -- update lockfiles with new requirements
## make requirements-update -- update lockfiles with newest versions for all dependencies
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

# Select a CUDA version. This influences the package index used to
# fetch binary wheels for PyTorch and related packages. When updating
# those packages, you may need to adjust this line.
CUDA_VERSION := $(if $(shell nvidia-smi --list-gpus 2> /dev/null),cu121,cpu)
CUDA_HOME := '' # dummy variable to make the CUDA_VERSION available in the environment

# select which requirements file to use
ifdef CI
REQUIREMENTS :=requirements-ci.lock
else
REQUIREMENTS :=requirements-all.lock
endif

help:
	@sed -nE 's/^## ?//p' $(MAKEFILE_LIST)

venv: $(VIRTUAL_ENV)/timestamp

$(VIRTUAL_ENV)/timestamp:
	pip install -q pip==24.0 pip-tools==7.4.0 uv==0.3.0
	# this installs all the dependencies as given in the lockfile
	echo "use $(REQUIREMENTS)"
	uv pip sync requirements/$(CUDA_VERSION)/$(REQUIREMENTS) --index-strategy unsafe-first-match
	# this sets the correct dependencies in pyproject.toml for the current platform
	python cereals.py set_platform $(CUDA_VERSION)
	# before installing the package itself, so the dependencies match what was installed
	# via the lockfile
	uv pip install -e .
	@touch $(VIRTUAL_ENV)/timestamp

format: venv
	isort src/encourage src/tests 
	ruff format src/encourage src/tests

lint: venv
	ruff check src/encourage src/tests 
	# we don't check the typing in the notebooks
	mypy src/encourage src/tests

tests: venv
	pytest

licenses: venv
	pip-licenses --from=mixed --order=license --summary

requirements:
	python cereals.py lock

requirements-update:
	python cereals.py lock --upgrade

.PHONY: help venv format lint tests licenses
