#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


# Generate OpenAI API
pushd $script_dir/..
rm -r openapi_server/* 2>/dev/null
mkdir openapi_server 2>/dev/null
curl --silent --output openapi_server/openapi.yaml https://raw.githubusercontent.com/openai/openai-openapi/a173760299883c3c8aa22ff700cc9b30c1b02003/openapi.yaml
git apply server/openapi.yaml.diff
openapi-generator-cli generate --input-spec openapi_server/openapi.yaml --generator-name python-fastapi --openapi-normalizer FILTER='operationId:createChatCompletion' --output generated-tmp/
cp -r generated-tmp/src/openapi_server/* openapi_server
rm -r generated-tmp
popd


# libCST
## Build libCST
pushd $script_dir/../lib/libCST
git remote add upstream git@github.com:Instagram/LibCST.git
git fetch --tags upstream
rm -rf ../../build/libCST 2>/dev/null
hatch run python -m build --sdist --outdir ../../build/libCST
popd

## Install libCST
pushd $script_dir/..
libcst_package=$(find build/libCST -type f)
pip install $libcst_package
popd


# CBMC
## Add missing libstdc++.so.6 softlink for cmake
pushd $CONDA_PREFIX/lib
libcstdcpp=$(find . -name 'libstdc++.so.6.*')
ln -s $(basename $libcstdcpp) libstdc++.so.6 2>/dev/null
popd

## Build CBMC
pushd $script_dir/../lib/cbmc
git submodule update --init
cmake -S . -Bbuild
cmake --build build --target cbmc --target goto-cc --target goto-instrument --parallel
popd

## Install CBMC
cp $script_dir/../lib/cbmc/build/bin/cbmc $CONDA_PREFIX/bin/
cp $script_dir/../lib/cbmc/build/bin/goto-cc $CONDA_PREFIX/bin/
cp $script_dir/../lib/cbmc/build/bin/goto-instrument $CONDA_PREFIX/bin/


# Z3
## Build and install Z3
pushd $script_dir/../lib/z3
git submodule update --init
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX -G "Unix Makefiles" ../
make --jobs
make install
popd


# Datasets
## Download FOLIO dataset
mkdir $script_dir/../datasets 2>/dev/null
pushd $script_dir/../datasets
hf download --repo-type dataset allenai/ZebraLogicBench-private grid_mode/test-00000-of-00001.parquet --local-dir .
hf download --repo-type dataset yale-nlp/FOLIO folio_v2_train.jsonl folio_v2_validation.jsonl --local-dir .
popd
