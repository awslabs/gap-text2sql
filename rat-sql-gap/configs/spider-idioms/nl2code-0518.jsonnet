# nl2code-0428-base with these modifications:
# - use start_lr of 2.5e-4
# (compare to bs=10,lr=2.5e-04,end_lr=0e0,att=0)

local _0428_base = import '../spider-20190205/nl2code-0428-base.libsonnet';
local output_from = false;

function(args) _0428_base(output_from=output_from) + {
    local ORIG_PREFIX = if 'data_path' in args then args.data_path else 'data/spider-20190205/',
    local PREFIX = ORIG_PREFIX + 
        '../spider-idioms/20190518/all-matches-trees-iter10-anysplit-att1/filt-none_st-%(st)s_nt-%(nt)d/' % args,

    model_name: 'filt-none_st-%(st)s_nt-%(nt)d' % args,

    data: {
        local data = self,
        train: {
            name: 'spider_idiom_ast',
            paths: [PREFIX + 'train.jsonl'],
            tables_paths: [ORIG_PREFIX + 'tables.json'],
            db_path: ORIG_PREFIX + 'database',
        },
        val: {
            name: 'spider_idiom_ast',
            paths: [PREFIX + 'val.jsonl'],
            tables_paths: data.train.tables_paths,
            db_path: data.train.db_path,
        },
    },

    model+: {
        decoder+: {
            multi_loss_type: 'mean',
        },
        encoder_preproc+: {
            save_path: PREFIX + 'nl2code',
        },
        decoder_preproc+: {
            grammar: {
                name: 'idiom_ast',
                base_grammar: {
                    name: 'spider',
                    output_from: output_from,
                    use_table_pointer: output_from,
                    include_literals: false,
                },
                template_file: PREFIX + 'templates.json',
                all_sections_rewritten: true,
                root_type: ['sql_plus_templates', 'sql'],
            },
        },
    },
    
    train+: {
        model_seed: args.att,
        data_seed: args.att,
        init_seed: args.att,
    },

    lr_scheduler+: {
        start_lr: 2.5e-4,
    },
}
