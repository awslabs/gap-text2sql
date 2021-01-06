{
    logdir: "logdir/arxiv",
    model_config: "configs/spider-20190205/arxiv-1906.11790v1.jsonnet",

    eval_name: "arxiv",
    eval_output: "ie_dir",
    eval_beam_size: 1,
    eval_steps: [40000],
    eval_section: "val",
}