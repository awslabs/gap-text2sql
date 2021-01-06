{
    local exp_id = 0,
    logdir: "/datadrive2/glove_run_%d" %exp_id,
    model_config: "configs/spider-20190205/nl2code-1204-glove.jsonnet",
    model_config_args: {
        att: 0,
        cv_link: false,
        clause_order: null, # strings like "SWGOIF"
        enumerate_order: true,
    },

    eval_name: "glove_run_%d_%s_%d" % [exp_id, self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}