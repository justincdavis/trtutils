.PHONY: help install clean docs test ci mypy pyright ruff release example-ci

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows"
	@echo "  mypy       to run the mypy static type checker"
	@echo "  pyright    to run the pyright static type checker"
	@echo "  ruff 	    to run ruff"
	@echo "  stubs      to generate the stubs"
	@echo "  test       to run the tests"
	@echo "  release    to perform all actions required for a release"
	@echo "  example-ci to run the CI workflows for the example scripts"

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
	python3 ci/build_example_docs.py
	sphinx-apidoc -o docs/source/ src/trtutils/ --separate --force
	cd docs && make html

blobs:
	python3 ci/compile_models.py --definitions

ci: ruff mypy

mypy:
	python3 -m mypy src/trtutils --config-file=pyproject.toml

pyright:
	python3 -m pyright --project=pyproject.toml

ruff:
	python3 -m ruff format ./src/trtutils
	python3 -m ruff check ./src/trtutils --fix --preview

stubs:
	python3 ci/make_stubs.py

test:
	./ci/run_tests.sh

example-ci:
	python3 -m ruff format ./examples
	python3 -m ruff check ./examples --fix --preview --ignore=T201,INP001,F841

release: clean ci test docs example-ci
