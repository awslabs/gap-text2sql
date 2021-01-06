# - output_from=true,false
# - qenc,ctenc=e,e;eb,ebs
# - upd_steps=3,4,5,6
# - max_steps=40k,80k
# - batch_size=10,20

local PREFIX = 'data/spider-20190205/';
local encoder_specs = {
    e: ['emb'],
    eb: ['emb', 'bilstm'],
    ebs: ['emb', 'bilstm-summarize'],
};

function(args) {
    model_name:
      'output_from=%(output_from)s,qenc=%(qenc)s,ctenc=%(ctenc)s,upd_steps=%(upd_steps)d,max_steps=%(max_steps)d,batch_size=%(batch_size)d' % args,

    data: {
        train: {
            name: 'spider', 
            paths: [
              PREFIX + 'train_%s.json' % [s]
              for s in ['spider', 'others']],
            tables_paths: [
              PREFIX + 'tables.json',
            ],
            db_path: PREFIX + 'database',
        },
        val: {
            name: 'spider', 
            paths: [PREFIX + 'dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'spiderv2',
            dropout: 0.2,
            word_emb_size: 256,
            question_encoder: encoder_specs[args.qenc],
            column_encoder: encoder_specs[args.ctenc],
            table_encoder: encoder_specs[args.ctenc],
            update_config: 
                if args.upd_steps == 0 then {
                    name: 'none',
                } else {
                    name: 'relational_transformer',
                    num_layers: args.upd_steps,
                    num_heads: 8,
                },
        },   
        decoder: {
            name: 'NL2Code',
            dropout: 0.2,
            desc_attn: 'mha',
        },
        encoder_preproc: {
            save_path: PREFIX + 'nl2code-0214-%s-%s/' % [
              if args.output_from then 'from' else 'nofrom',
              'notinc',
              ],
            min_freq: 3,
            max_count: 5000,
            include_table_name_in_column: false,#args.tinc,
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'spider',
                output_from: args.output_from,
                use_table_pointer: args.output_from,
                include_literals: false,
            },
            use_seq_elem_rules: true,
            include_table_name_in_column:: false,
        },
    },

    train: {
        batch_size: args.batch_size,
        eval_batch_size: 50,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: args.max_steps,
        num_eval_items: 50,
    },
    optimizer: {
        name: 'adam',
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
