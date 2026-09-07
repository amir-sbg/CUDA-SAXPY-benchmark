.PHONY: configure build run sweep clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --config Release --parallel

run: build
	./build/cuda_saxpy

sweep: build
	python3 scripts/run_block_sweep.py --output reports/block_sweep.csv

clean:
	rm -rf build
