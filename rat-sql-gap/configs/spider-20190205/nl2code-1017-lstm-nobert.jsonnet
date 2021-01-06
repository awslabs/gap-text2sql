local _0428_base = import 'nl2code-0428-base.libsonnet';
local _data_path = 'data/spider-20190205/';

_0428_base(output_from=true, data_path=_data_path) + {
    local lr = 0.000743552663260837,
    local end_lr = 0,
    local bs = 20,
    local att = 0,

    local lr_s = '%0.1e' % lr,
    local end_lr_s = '0e0',
    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,att=%(att)d' % ({
        bs: bs,
        lr: lr_s,
        end_lr: end_lr_s,
        att: att,
    }),

    model+: {
        encoder+: {
            batch_encs_update: false,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config+:  {
                name: 'none',
            },
            top_k_learnable: 50,
        },
        encoder_preproc+: {
            word_emb+: {
                name: 'glove',
                kind: '42B',
                lemmatize: true,
            },
            min_freq: 4,
            max_count: 5000,
            compute_sc_link: true,
            count_tokens_in_word_emb_for_vocab: true,
            save_path: _data_path + 'nl2code-iclr-0925',
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: true,
            },
            save_path: _data_path + 'nl2code-iclr-0925',

            compute_sc_link :: null,
        },
        decoder+: {
            name: 'NL2Code',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            recurrent_size : 512,
            loss_type: "softmax",
            use_align_mat: false,
            use_align_loss: false,
        },
    },

    train+: {
        batch_size: bs,

        model_seed: att,
        data_seed:  att,
        init_seed:  att,
    },

    lr_scheduler+: {
        start_lr: lr,
        end_lr: end_lr,
    },

}
