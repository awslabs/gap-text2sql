#!/bin/bash

pip install --user -e .
pip install --user torch==1.3.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user entmax
sudo apt-get update
sudo apt-get install -y default-jre
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
export PT_OUTPUT_DIR="${PT_OUTPUT_DIR:-$PWD}"
export PT_DATA_DIR="${PT_DATA_DIR:-$PWD}"

export CORENLP_HOME="$PT_DATA_DIR/stanford-corenlp-full-2018-10-05/"
export CACHE_DIR=${PT_DATA_DIR}
export XDG_CACHE_HOME=${PT_DATA_DIR}

export LC_ALL="C.UTF-8"
export LANG="C.UTF-8"
