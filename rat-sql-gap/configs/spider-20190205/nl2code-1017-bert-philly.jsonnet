local _0428_base = import 'nl2code-0428-base.libsonnet';
local _output_from = true;
local _fs = 2;

function(args) _0428_base(output_from=_output_from, data_path=args.data_path) + {
    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s,att=%(att)d' % (args + {
        lr: lr_s,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        encoder+: {
            name: 'spider-bert',
            batch_encs_update:: null,
            question_encoder:: null,
            column_encoder:: null,
            table_encoder:: null,
            dropout:: null,
            update_config+:  {
                name: 'relational_transformer',
                num_layers: 8,
                num_heads: 8,
                sc_link: args.sc_link,
            },
            bert_token_type: args.bert_token_type,
            top_k_learnable:: null,
            word_emb_size:: null,
        },
        encoder_preproc+: {
            word_emb:: null,
            min_freq:: null,
            max_count:: null,
            compute_sc_link: true,
            fix_issue_16_primary_keys: true,
            count_tokens_in_word_emb_for_vocab:: null,
            save_path: args.data_path + 'nl2code-1006,output_from=%s,fs=%d,emb=bert' % [_output_from, _fs],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: args.end_with_from,
                clause_order: args.clause_order,
                infer_from_conditions: true,
                factorize_sketch: _fs,
            },
            save_path: args.data_path + 'nl2code-1006,output_from=%s,fs=%d,emb=bert' % [_output_from, _fs],

            compute_sc_link:: null,
            fix_issue_16_primary_keys:: null,
        },
        decoder+: {
            name: 'NL2Code',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: 768,
            recurrent_size : args.decoder_hidden_size,
            loss_type: 'softmax',
            use_align_mat: args.use_align_mat,
            use_align_loss: args.use_align_loss,
        },

        log: {
            reopen_to_flush:true,
        },
    },

    train+: {
        batch_size: args.bs,
        num_batch_accumulated: args.num_batch_accumulated,
        clip_grad: 1,
        keep_every_n: 2000,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },

    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: args.bert_lr,
    },

    lr_scheduler+: {
        name: 'warmup_polynomial_group',
        start_lrs: [args.lr, args.bert_lr],
        end_lr: args.end_lr,
    },

}
