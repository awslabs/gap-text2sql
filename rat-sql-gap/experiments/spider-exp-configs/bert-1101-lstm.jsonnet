{
    logdir: "/datadrive1/bert_run9",
    model_config: "configs/spider-20190205/nl2code-1017-lstm-bert.jsonnet",
    model_config_args: {
        bs: 8,
        num_batch_accumulated: 4,
        lr: 7.44e-4,
        bert_lr: 3e-6,
        att: 0,
        end_lr: 0,
        sc_link: true,
        use_align_mat: false,
        use_align_loss: false,
        bert_token_type: false,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: "SWGOIF", # strings like "SWGOIF", it will be prioriotized over end_with_from
    },

    eval_name: "bert_run9_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: false,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}