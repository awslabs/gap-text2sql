{
    logdir: "logdir/bert_run3",
    model_config: "configs/spider-20190205/nl2code-1001-bert.jsonnet",
    model_config_args: {
        bs: 8,
        lr: 7.44e-4,
        bert_lr: 3e-6,
        att: 0,
        end_lr: 0
    },

    eval_name: "bert_run3",
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}