# Things to vary
# - output_from: whether to output the 'from' portion within the model
# - question_encoder
#   - emb
#   - emb and bilstm
# - column_encoder
#   - emb
#s   - emb and bilstm
#   - emb and bilstm-summarize
# - table_encoder
#   - emb
#   - emb and bilstm
#   - emb and bilstm-summarize
# - update_config
#   - [for future] type
#   - [for future] whether to mask updates between the same entity
#   - number of update steps
#   - [for future] relations to use
#
# x what to include in description memory: everything
# x use multi-head attention in decoder: yes
# x whether model should be allowed to point to a column for a table: yes
# x whether to exclude literals: yes
#
# output_from=yes,no
# qenc=e,eb
# ctenc=e,eb,ebs
# tinc=yes,no
# upd_steps=0,1,2,3

local PREFIX = 'data/spider-20190205/';
local encoder_specs = {
    e: ['emb'],
    eb: ['emb', 'bilstm'],
    ebs: ['emb', 'bilstm-summarize'],
};

function(args) {
    model_name:
      'output_from=%(output_from)s,qenc=%(qenc)s,ctenc=%(ctenc)s,tinc=%(tinc)s,upd_steps=%(upd_steps)d' % args,

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
              if args.tinc then 'tinc' else 'notinc',
              ],
            min_freq: 3,
            max_count: 5000,
            include_table_name_in_column: args.tinc,
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
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: 2000,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: 40000 - self.num_warmup_steps,
        power: 0.5,
    }
}
