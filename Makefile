# Thin wrapper over CMake presets. Everything here is one line of cmake; the
# Makefile exists so that "make" and "make test" do the obvious thing in a
# clean container, which is what the evaluator will type first.
#
# Real build documentation: docs/build.md

PRESET ?= dev
BUILD  := build/$(PRESET)
JOBS   ?= $(shell nproc)

.DEFAULT_GOAL := build

.PHONY: help configure build test test-unit test-e2e bench release asan tsan \
        fixtures fixtures-small format format-check tidy clean distclean docs demo

help:                     ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

configure:                ## Configure with $(PRESET)
	cmake --preset $(PRESET)

build: configure          ## Build with $(PRESET)
	cmake --build --preset $(PRESET) -j $(JOBS)

test: build               ## Build then run the full test suite
	ctest --preset $(PRESET)

test-unit: build          ## Unit tests only (fast; no fixtures needed)
	ctest --preset unit

test-e2e: build           ## End-to-end suite (needs fixtures)
	ctest --preset e2e

bench: ## Build the release preset and run every benchmark
	cmake --preset release && cmake --build --preset release -j $(JOBS)
	./bench/scripts/run_all.sh build/release

release:                  ## Optimised build, the one benchmarks must come from
	$(MAKE) PRESET=release build

asan:                     ## Address+UB sanitizer build and tests
	$(MAKE) PRESET=asan test

tsan:                     ## Thread sanitizer, concurrency tests only
	cmake --preset tsan && cmake --build --preset tsan -j $(JOBS) && ctest --preset tsan

fixtures-small:           ## Generate the MLP + small CNN fixtures (~200 MB)
	python3 -m venv fixtures/.venv
	fixtures/.venv/bin/pip install -q -r fixtures/requirements.txt
	fixtures/.venv/bin/python fixtures/gen_mlp.py    --out fixtures/out
	fixtures/.venv/bin/python fixtures/gen_resnet.py --out fixtures/out

fixtures: fixtures-small  ## Generate all fixtures including the 7B pair (large, slow)
	fixtures/.venv/bin/python fixtures/gen_7b.py --out fixtures/out

format:                   ## Reformat all C/C++ sources in place
	@git ls-files '*.cpp' '*.hpp' '*.c' '*.h' | xargs clang-format -i

format-check:             ## Fail if anything is unformatted (what CI runs)
	@git ls-files '*.cpp' '*.hpp' '*.c' '*.h' | xargs clang-format --dry-run --Werror

tidy: configure           ## clang-tidy over the compile database
	run-clang-tidy -p $(BUILD) -quiet modules apps

demo:                     ## Scripted end-to-end demo (used in the presentation)
	./scripts/demo.sh

clean:                    ## Remove the current preset's build tree
	rm -rf $(BUILD)

distclean:                ## Remove all build output and generated fixtures
	rm -rf build install vcpkg_installed fixtures/out fixtures/.venv
