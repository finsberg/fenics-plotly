.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

export FENICS_PLOTLY_RENDERER=notebook

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	python3 -m flake8 fenics_plotly tests

type: ## Run mypy
	python3 -m mypy fenics_plotly tests

test: ## run tests on every Python version with tox
	python3 -m pytest

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/fenics_plotly.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs fenics_plotly
	for file in README.md CONTRIBUTING.md INSTALL.md AUTHORS.md; do \
		cp $$file docs/. ;\
	done
	jupyter-book build docs
	# python -m http.server --directory docs/_build/html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	python3 -m twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} dist/*

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python3 -m pip install .

dev: clean ## Just need to make sure that libfiles remains
	python3 -m pip install -r requirements_dev.txt
	python3 -m pip install -e .
	pre-commit install

bump:
	bump2version patch
