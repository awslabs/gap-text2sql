{
    logdir: "/datadrive/bert_adapt/bert_run8",
    model_config: "configs/spider-20190205/nl2code-1017-lstm-nobert.jsonnet",

    eval_name: "bert_run8",
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: false,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}
