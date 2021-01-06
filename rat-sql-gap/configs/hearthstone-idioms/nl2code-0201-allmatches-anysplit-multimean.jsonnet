function(args) {
  local PREFIX = 'data/hearthstone-idioms-20190201/all-matches-trees-anysplit/filt-%(filt)s_st-%(st)s_nt-%(nt)d/' % args,

    data: {
        train: {
            name: 'idiom_ast', 
            path: PREFIX + 'train.jsonl',
        },
        val: {
            name: 'idiom_ast', 
            path: PREFIX + 'dev.jsonl',
        },
        test: {
            name: 'idiom_ast', 
            path: PREFIX + 'test.jsonl',
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'NL2Code',
            dropout: 0.2,
        },   
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
            multi_loss_type: 'mean',
        },
        encoder_preproc: {
            save_path: PREFIX + 'nl2code/',
            min_freq: 3,
            max_count: 5000,
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'idiom_ast',
                base_grammar: {
                    name: 'python',
                },
                template_file: PREFIX + 'templates.json',
                all_sections_rewritten: true,
            },
            use_seq_elem_rules: true,
        },
    },

    train: {
        batch_size: 10,
        eval_batch_size: self.batch_size,

        keep_every_n: 100,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: 2650,
        num_eval_items: 66,

        eval_on_train: true,
        eval_on_val: false,
    },

    optimizer: {
        name: 'adadelta',
        lr: 1.0,
        rho: 0.95,
        eps: 1e-6,
    },
}
