local PREFIX = 'data/spider-20181217/';
{
    data: {
        train: {
            name: 'spider', 
            paths: [PREFIX + 'train_%s.json' % [s] for s in ['spider', 'others']],
            tables_paths: [PREFIX + 'tables.json'],
        },
        val: {
            name: 'spider', 
            paths: [PREFIX + 'dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'spider',
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
              name: 'spider',
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
        num_eval_items: 50,
    },
    optimizer: {
        name: 'adam',
    },

}
