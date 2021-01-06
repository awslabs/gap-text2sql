local PREFIX = 'data/spider-20190205/';

function (args) {
    data: {
        train: {
            name: 'spider', 
            paths: [
              PREFIX + 'train_%s.json' % [s]
              for s in ['spider', 'others'] + (if args.augment then ['wikisql_augment'] else [])],
            tables_paths: [
              PREFIX + 'tables.json',
            ] + (
              if args.augment then [PREFIX + 'tables_wikisql_augment.json'] else []
            ),
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
            table_enc: if args.output_from then 'mean_columns' else 'none',
        },   
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
        },
        encoder_preproc: {
            save_path: PREFIX + 'nl2code-0206-%s-%s/' % [
              if args.augment then 'aug' else 'noaug',
              if args.output_from then 'from' else 'nofrom'],
            min_freq: 3,
            max_count: 5000,
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
              name: 'spider',
              output_from: args.output_from,
              use_table_pointer: args.output_from,
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
