dist: ## build source and wheel package
	python3 setup.py sdist bdist_wheel

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/

test: ## run tests quickly with the default Python
	pytest

release: ## package and upload release
	twine upload dist/*

release-test: ## package and upload release on test PyPI
	twine upload --repository testpypi dist/*
