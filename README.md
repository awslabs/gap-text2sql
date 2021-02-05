# GAP-text2SQL: Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training

Code and model from our [AAAI 2021 paper](https://arxiv.org/abs/2012.10309)

## Updates

[2020/02/05] Support to run the model on own databases and queries. Check out the [notebook](rat-sql-gap/notebook.ipynb). 

## Abstract

Most recently, there has been significant interest in learning contextual representations for various NLP tasks, by leveraging large scale text corpora to train large neural language models with self-supervised learning objectives, such as Masked Language Model (MLM). However, based on a pilot study, we observe three issues of existing general-purpose language models when they are applied to text-to-SQL semantic parsers: fail to detect column mentions in the utterances, fail to infer column mentions from cell values, and fail to compose complex SQL queries. To mitigate these issues, we present a model pre-training framework, Generation-Augmented Pre-training (GAP), that jointly learns representations of natural language utterances and table schemas by leveraging generation models to generate pre-train data. GAP MODEL is trained on 2M utterance-schema pairs and 30K utterance-schema-SQL triples, whose utterances are produced by generative models. Based on experimental results, neural semantic parsers that leverage GAP MODEL as a representation encoder obtain new state-of-the-art results on both SPIDER and CRITERIA-TO-SQL benchmarks.

## Setup
```bash
conda create --name gap-text2sql python=3.7
source activate gap-text2sql
conda install pytorch=1.5 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Download the dataset
```bash
pip install gdown
cd rat-sql-gap
gdown --id 1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0
unzip spider.zip
bash data/spider/generate.sh ./spider
```

### Build dataset directory
```bash
mkdir data/spider-bart
cp ./spider/tables.json data/spider-bart/
cp ./spider/train_spider.json data/spider-bart/
cp ./spider/train_others.json data/spider-bart/
cp ./spider/dev.json data/spider-bart/
ln -s $(pwd)/spider/database data/spider-bart/database
```

### Download the library
```bash
mkdir third_party
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip -d third_party/
```

### Start the Stanford library
```bash
pushd third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8999 -timeout 15000 > server.log &
popd
```

### Download the checkpoint
```bash
mkdir -p logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/
mkdir ie_dirs
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/gap-finetuned-checkpoint logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/model_checkpoint-00041000

mkdir -p pretrained_checkpoint
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/pretrained-checkpoint pretrained_checkpoint/pytorch_model.bin
```

Alternatively, you can download them here if you don't have awscli:
[gap-finetuned-checkpoint](https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/gap-finetuned-checkpoint)
and [pretrained-checkpoint](https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/pretrained-checkpoint)

```bash
curl https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/gap-finetuned-checkpoint -o logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/model_checkpoint-00041000
curl https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/pretrained-checkpoint -o pretrained_checkpoint/pytorch_model.bin
```


### Preprocess dataset
```bash
python run.py preprocess experiments/spider-configs/gap-run.jsonnet
```

## Inference
```bash
python run.py eval experiments/spider-configs/gap-run.jsonnet
```

You then get the inference results and evaluation results in the paths:`ie_dirs/bart_run_1_true_1-step41000.infer` and `ie_dirs/bart_run_1_true_1-step41000.eval`.

## Training

```bash
python run.py train experiments/spider-configs/gap-run.jsonnet
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
