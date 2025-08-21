# MCP Testing Makefile

.PHONY: help test test-unit test-integration test-performance test-all coverage clean

help:
	@echo "Available commands:"
	@echo "  make test-unit       - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-performance - Run performance tests"
	@echo "  make test-all        - Run all tests"
	@echo "  make coverage        - Run tests with coverage"
	@echo "  make clean          - Clean test artifacts"

test-unit:
	python run_tests.py unit

test-integration:
	python run_tests.py integration

test-performance:
	python run_tests.py performance

test-all:
	python run_tests.py all

coverage:
	python run_tests.py all --coverage

smoke:
	python run_tests.py smoke

ci:
	python run_tests.py ci

clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf test-results.xml
	rm -rf ci-test-report.json
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

install-deps:
	pip install -r requirements-test.txt

setup:
	python setup_testing.py
