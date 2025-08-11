.PHONY: help install clean docs test ci mypy pyright ruff format check release ci_env mypy_venv venv_format venv_check civ ruff_venv act

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows - mypy & ruff"
	@echo "  mypy       to run the mypy static type checker"
	@echo "  pyright    to run the pyright static type checker"
	@echo "  format     to run the ruff formatter"
	@echo "  check      to run the ruff linter"
	@echo "  ruff 	    to run both the formatter and linter from ruff"
	@echo "  mypy_venv  to run the mypy static type checker in the CI environment"
	@echo "  civ    to run the CI workflows in the CI environment"
	@echo "  ruff_venv  to run both the formatter and linter from ruff in the CI environment"
	@echo "  stubs      to generate the stubs"
	@echo "  test       to run the tests"
	@echo "  release    to perform all actions required for a release"
	@echo "  act        to run all GitHub Actions workflows locally with act (push event)"

install:
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf trtutils/*.egg-info
	rm -rf src/trtutils/*.egg-info
	pyclean .
	rm -rf .mypy_cache
	rm -rf .ruff_cache

docs:
	rm -rf docs/_build/*
	python3 ci/build_benchmark_docs.py
	python3 ci/build_example_docs.py
	sphinx-apidoc -o docs/source/ src/trtutils/ --separate --force
	cd docs && make html

pyright:
	python3 -m pyright --project=pyproject.toml

stubs:
	python3 ci/make_stubs.py

test:
	./ci/run_tests.sh

ci: ruff mypy

ruff: format check

mypy:
	python3 -m mypy examples --config-file=pyproject.toml
	python3 -m mypy tests --config-file=pyproject.toml
	python3 -m mypy src/trtutils --config-file=pyproject.toml

format:
	python3 -m ruff format ./demos
	python3 -m ruff format ./examples
	python3 -m ruff format ./tests
	python3 -m ruff format ./src/trtutils

check:
	python3 -m ruff check ./demos --fix --preview --ignore=INP001,T201
	python3 -m ruff check ./examples --fix --preview --ignore=INP001,T201,D103
	python3 -m ruff check ./tests --fix --preview --ignore=S101,D100,D104,PLR2004,T201
	python3 -m ruff check ./src/trtutils --fix --preview

release: clean ci test docs

ci_env:
	uv venv .venv-ci --python 3.9
	. .venv-ci/bin/activate && \
	uv pip install ".[all]" ".[ci]" ".[test]"

mypy_venv: ci_env
	. .venv-ci/bin/activate && \
	$(MAKE) mypy

venv_format: ci_env
	. .venv-ci/bin/activate && \
	$(MAKE) format

venv_check: ci_env
	. .venv-ci/bin/activate && \
	$(MAKE) check

civ: ruff_venv mypy_venv

ruff_venv: venv_format venv_check

act:
	act $(ACT_ARGS)
