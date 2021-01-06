# Re-do of 0919 and 0923 ablations.
# args:
# - align_mat (bool): use separate alignment matrix for columns/tables
# - align_loss (bool): a separate alignment loss on top
# - schema_link (bool): whether to use schema linking feature in encoder
# - fix_primary_keys (bool): fix primary keys (rshin/seq2struct#16)
# - merge_types (bool): use encoder in boolean
# - att (int): attempt number

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
    model_name: 'am=%(align_mat)s,al=%(align_loss)s,sl=%(schema_link)s,fpk=%(fix_primary_keys)s,mt=%(merge_types)s,att=%(att)d' % args,

    local preproc_path = args.data_path + 'nl2code-0924-ablations,sl=%(schema_link)s,fpk=%(fix_primary_keys)s' % args,

    model+: {
        encoder+: {
            update_config: super.update_config + (
                if args.merge_types then {
                    # Get rid of all relation types except qq_dist
                    cc_foreign_key: false,
                    cc_table_match: false,
                    ct_foreign_key: false,
                    ct_table_match: false,
                    tc_foreign_key: false,
                    tc_table_match: false,
                    tt_foreign_key: false,
                    merge_types: true,
                } else {}
            ),
        },

        encoder_preproc+: {
            save_path: preproc_path,
            compute_sc_link: args.schema_link,
            fix_issue_16_primary_keys: args.fix_primary_keys,
        },
        decoder_preproc+: {
            save_path: preproc_path,
            compute_sc_link:: null,
            fix_issue_16_primary_keys:: null,
        },
        decoder+: {
            use_align_mat: args.align_mat,
            use_align_loss: args.align_loss,
        },
    },
}
