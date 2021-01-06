# Base:
# {glove: true, upd_type: 'full', num_layers: 4}

# Ablations
#   no word embeddings
#   {glove: false, upd_type: 'full', num_layers: 4, att: __ATT__}
#
#   upd_types
#   {glove: true, upd_type: 'no_subtypes', num_layers: 4, att: __ATT__}
#   {glove: true, upd_type: 'merge_types', num_layers: 4, att: __ATT__}
#
#   fewer updates
#   {glove: true, upd_type: 'full', num_layers: 2, att: __ATT__}
#   {glove: true, upd_type: 'full', num_layers: 0, att: __ATT__}

local _0428_base = import 'nl2code-0428-base.libsonnet';
local PREFIX = 'data/spider-20190205/';

local enc_update_types = {
    full: {},
    no_subtypes: {
        cc_foreign_key: false,
        cc_table_match: false,
        ct_foreign_key: false,
        ct_table_match: false,
        tc_foreign_key: false,
        tc_table_match: false,
        tt_foreign_key: false,
    },
    merge_types: self.no_subtypes {
        merge_types: true,
    }
};

function(args) _0428_base(output_from=false) + {
    model_name: 'glove=%(glove)s,upd_type=%(upd_type)s,num_layers=%(num_layers)d,att=%(att)d' % args,

    model+: {
        encoder+: {
            batch_encs_update: args.num_layers == 0,
            update_config: if args.num_layers == 0 then {
                name: 'none',
            } else super.update_config + {
                num_layers: args.num_layers,
            } + enc_update_types[args.upd_type],
        },

        encoder_preproc+: {
            word_emb: if args.glove then super.word_emb else null,
            count_tokens_in_word_emb_for_vocab: !args.glove,
            min_freq: if args.glove then super.min_freq else 3,
            save_path: PREFIX + 'nl2code-0521,glove=%(glove)s/' % args,
        },
    },

    train+: {
        batch_size: 50,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },
}
