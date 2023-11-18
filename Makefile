.PHONY: help install clean docs blobs test ci mypy

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows"
	@echo "  mypy       to run the type checker"
	@echo "  test       to run the tests"

install:
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf trtutils/*.egg-info
	rm -rf src/trtutils/*.egg-info
	pyclean .
	cd docs && make clean

docs:
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/trtutils/
	cd docs && make html

ci: mypy
	./scripts/ci/pyupgrade.sh
	python3 -m ruff ./src//trtutils --fix
	python3 -m isort src/trtutils
	python3 -m black src/trtutils --safe

mypy:
	./scripts/ci/mypy.sh

test:
	./scripts/run_tests.sh
