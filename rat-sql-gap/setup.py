from setuptools import setup, find_packages

setup(
    name='seq2struct',
    version='0.0.1',
    description='seq2struct',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    package_data={
        '': ['*.asdl'],
    },
    install_requires=[
        'asdl~=0.1.5',
        'babel~=2.7.0',
        'astor~=0.7.1',
        'attrs~=18.2.0',
        'bpemb~=0.2.11',
        'cython~=0.29.1',
        'jsonnet~=0.11.2',
        'nltk~=3.4',
        'networkx~=2.2',
        'numpy~=1.16',
        'pyrsistent~=0.14.9',
        'records~=0.5.3',
        'stanford-corenlp~=3.9.2',
        'tabulate~=0.8.6',
        'torch==1.3.1',
        'torchtext~=0.3.1',
        'tqdm~=4.36.1',
        'transformers~=2.1.1',
    ],
    entry_points={"console_scripts": ["seq2struct=run:main"]},
)
