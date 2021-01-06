function(args) {
  local PREFIX =
  'data/django-idioms-20190403/rewritten-trees-iter10-anysplit-att1/filt-%(filt)s_st-%(st)s_nt-%(nt)d/' % args,

    data: {
        train: {
            name: 'idiom_ast', 
            path: PREFIX + 'train.jsonl',
        },
        val: {
            name: 'idiom_ast', 
            path: PREFIX + 'dev.jsonl',
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
                root_type: 'stmt',
            },
            use_seq_elem_rules: true,
        },
    },

    train: {
        batch_size: 10,
        eval_batch_size: self.batch_size,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: 40000,
        num_eval_items: 100,

        eval_on_train: true,
        eval_on_val: true,
    },

    optimizer: {
        name: 'adam',
    },
}
