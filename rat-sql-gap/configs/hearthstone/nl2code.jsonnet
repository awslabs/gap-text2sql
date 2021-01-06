local PREFIX = 'data/hearthstone/';
{
    data: {
        train: {
            name: 'hearthstone', 
            path: PREFIX + 'train_hs',
        },
        val: {
            name: 'hearthstone', 
            path: PREFIX + 'dev_hs',
        },
        test: {
            name: 'hearthstone', 
            path: PREFIX + 'test_hs',
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

        max_steps: 2650,
        num_eval_items: 66,
    },
    optimizer: {
        name: 'adadelta',
        lr: 1.0,
        rho: 0.95,
        eps: 1e-6,
    },
}
