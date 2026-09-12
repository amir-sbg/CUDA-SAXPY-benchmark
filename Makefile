.PHONY: configure build run sweep summarize clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --config Release --parallel

run: build
	./build/cuda_saxpy

sweep: build
	python3 scripts/run_block_sweep.py --output reports/block_sweep.csv

summarize:
	python3 scripts/summarize_sweep.py --input reports/block_sweep.csv --output reports/block_sweep.md

clean:
	rm -rf build
