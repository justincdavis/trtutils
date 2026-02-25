.PHONY: help clean benchmark-bootstrap docs fix ci ci_env typecheck test test-legacy test-new test-cpu test-cov test-cov-html release docker

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean               to clean the directory tree"
	@echo "  benchmark-bootstrap to download all benchmark models upfront"
	@echo "  docs                to generate the documentation"
	@echo "  fix                 to auto-fix formatting and lint issues"
	@echo "  ci                  to run GitHub Actions workflows locally (act or gh act)"
	@echo "  ci_env              to create the CI environment"
	@echo "  typecheck           to run the ty static type checker"
	@echo "  test                to run the tests"
	@echo "  test-legacy         to run legacy tests only"
	@echo "  test-new            to run new tests only (ignoring legacy)"
	@echo "  test-cpu            to run CPU-only tests"
	@echo "  test-cov            to run tests with coverage (JSON report)"
	@echo "  test-cov-html       to run tests with coverage (HTML report)"
	@echo "  release             to perform all actions required for a release"
	@echo "  docker              to build the local act Docker image"

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf trtutils/*.egg-info
	rm -rf src/trtutils/*.egg-info
	pyclean .
	rm -rf .ruff_cache

benchmark-bootstrap:
	python3 benchmark/run.py --bootstrap --model all

docs:
	./ci/make_docs.sh

test:
	./ci/run_tests.sh

test-legacy:
	python3 -m pytest -rP -v tests/legacy/

test-new:
	python3 -m pytest -rP -v tests/ --ignore=tests/legacy/

test-cpu:
	python3 -m pytest -rP -v tests/ --ignore=tests/legacy/ -m cpu

test-cov:
	python3 -m pytest --cov=src/trtutils --cov-branch --cov-report=term-missing --cov-report=json:.coverage.json tests/ --ignore=tests/legacy/

test-cov-html:
	python3 -m pytest --cov=src/trtutils --cov-branch --cov-report=html --cov-report=json:.coverage.json tests/ --ignore=tests/legacy/

fix:
	./ci/run_format.sh
	./ci/run_lint.sh

ci:
	./ci/act.sh $(ACT_ARGS)

typecheck:
	./ci/run_ci.sh --typecheck

release: clean ci test docs

ci_env:
	./ci/make_venv.sh

docker:
	docker build -f docker/Dockerfile.act -t trtutils-act:latest .
