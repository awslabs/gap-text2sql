# Ablations with very few realtion types.

local _0715_align_philly = import 'nl2code-0715-align-philly.jsonnet';

function(args) _0715_align_philly(args + { 
    end_lr: 0,
    setting: 'basic',
    loss_type: 'softmax',

    # Copied from application_1568109446035_7203
    # in /mnt/wuphillyblob_share/spider/spidersweep-rr2/philly.yaml
    # (//wuphillyblob.file.core.windows.net/phillytools/)
    bs: 20,
    lr: 0.000743552663260837,
    top_k_learnable: 50,
    decoder_recurrent_size: 512,
    decoder_dropout: 0.20687225956012834,
    num_layers: 8,
}) + {
    model_name: 'no_rels,am=%(align_mat)s,al=%(align_loss)s,att=%(att)d' % args,

    local preproc_path = (
      args.data_path +
      'basic,nl2code-0815,output_from=true,emb=glove-42B,min_freq=4/'
    ),

    model+: {
        encoder+: {
            update_config+: {
                # Get rid of all relation types except qq_dist
                cc_foreign_key: false,
                cc_table_match: false,
                ct_foreign_key: false,
                ct_table_match: false,
                tc_foreign_key: false,
                tc_table_match: false,
                tt_foreign_key: false,
                merge_types: true,
            },
        },

        encoder_preproc+: {
            save_path: preproc_path,
        },
        decoder_preproc+: {
            save_path: preproc_path,
        },
        decoder+: {
            use_align_mat: args.align_mat,
            use_align_loss: args.align_loss,
        },
    },
}
