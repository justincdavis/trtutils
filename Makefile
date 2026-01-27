.PHONY: help clean benchmark-bootstrap docs ci ci_env format lint typecheck test release act docker

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean               to clean the directory tree"
	@echo "  benchmark-bootstrap to download all benchmark models upfront"
	@echo "  docs                to generate the documentation"
	@echo "  ci 	             to run the CI workflows - ty & ruff"
	@echo "  ci_env              to create the CI environment"
	@echo "  format              to run the ruff formatter"
	@echo "  lint                to run the ruff linter"
	@echo "  typecheck           to run the ty static type checker"
	@echo "  test                to run the tests"
	@echo "  release             to perform all actions required for a release"
	@echo "  docker              to build the local act Docker image"
	@echo "  act                 to run all GitHub Actions workflows locally with act (push event)"

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

ci:
	./ci/run_ci.sh --format --lint --typecheck

ruff: format lint

typecheck:
	./ci/run_ci.sh --typecheck

format:
	./ci/run_ci.sh --format

lint:
	./ci/run_ci.sh --lint

release: clean ci test docs

ci_env:
	./ci/make_venv.sh

docker:
	docker build -f docker/Dockerfile.act -t trtutils-act:latest .

act:
	act $(ACT_ARGS)
