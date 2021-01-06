local _0428_base = import 'nl2code-0428-base.libsonnet';
local _data_path = 'data/spider-20190205/';
local _output_from = true;
local _min_freq = 4;

function(args) _0428_base(output_from=_output_from, data_path=_data_path) + {
    local lr_s = '%0.1e' % args.lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,att=%(att)d' % (args + {
        lr: lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        encoder+: {
            name: 'spiderv2',
            batch_encs_update: false,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config+:  {
                name: 'relational_transformer',
                num_layers: args.num_layers,
                num_heads: 8,
                sc_link: true,
            },
            top_k_learnable: args.top_k_learnable,
        },
        encoder_preproc+: {
            word_emb+: {
                name: 'glove',
                kind: '42B',
                lemmatize: true,
            },
            min_freq: _min_freq,
            max_count: 5000,
            count_tokens_in_word_emb_for_vocab: true,
            save_path: _data_path + 'basic,nl2code-0815,output_from=%s,emb=glove-42B,min_freq=%s/' % [_output_from, _min_freq],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: true,
            },
            save_path: _data_path + 'basic,nl2code-0815,output_from=%s,emb=glove-42B,min_freq=%s/' % [_output_from, _min_freq],
        },
        decoder+: {
            name: 'NL2Code',
            dropout: args.decoder_dropout,
            desc_attn: 'mha',
            recurrent_size : args.decoder_recurrent_size,
            loss_type: args.loss_type,
            use_align_mat: true,
            use_align_loss: true,
        },
    },

    train+: {
        batch_size: args.bs,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },

    lr_scheduler+: {
        start_lr: args.lr,
        end_lr: args.end_lr,
    },

}
