PY ?= python
PYTHONPATH := $(CURDIR)/python
export PYTHONPATH

.PHONY: build test test-cpp test-py bench bench-cpp demo clean

build:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build build

test-cpp: build
	./build/hg_tests

test-py: build
	$(PY) -m pytest -q

test: test-cpp test-py

bench: build
	$(PY) -m hackablegrid.bench

bench-cpp: build
	./build/hg_bench

demo: build
	$(PY) -m hackablegrid.demo

clean:
	rm -rf build results python/hackablegrid/*.so
