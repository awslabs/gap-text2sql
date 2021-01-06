local PREFIX = 'data/django/';
{
    data: {
        train: {
            name: 'django', 
            path: PREFIX + 'train.jsonl',
        },
        val: {
            name: 'django', 
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
            min_freq: 5,
            max_count: 5000,
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'python',
            },
        },
    },

    train: {
        batch_size: 10,
        eval_batch_size: self.batch_size,

        keep_every_n: 100,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: 80000,
        num_eval_items: 100,
    },
    optimizer: {
        name: 'adam',
    },
}
