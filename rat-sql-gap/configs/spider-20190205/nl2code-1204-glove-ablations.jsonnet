# Remove schema edges: foreign key, primary key, which column is in which table
# -> schema_edges=False
# Remove schema linking: n-gram matching between columns and tables
# -> schema_link=False
# Remove alignment loss: keeps the matrix between question words and columns, but removes the objective
# -> align_mat=False
# Remove alignment loss + alignment matrix
# -> align_mat=False
# -> align_loss=False
# Remove everything above (but keep the Transformer)

local _1204_glove = import 'nl2code-1204-glove.jsonnet';

function(args) _1204_glove(args + {cv_link: false, clause_order: null, enumerate_order: false}, data_path=args.data_path) + {
    local preproc_path = args.data_path + 'nl2code-1204-glove-ablations,sc_link=%s' % args.schema_link,
    model_name: 'se=%(schema_edges)s,sl=%(schema_link)s,am=%(align_mat)s,al=%(align_loss)s,att=%(att)d' % args,

    model+: {
        encoder+: {
            update_config+: ((
                if args.schema_edges then {} else {
                    cc_foreign_key: false,
                    cc_table_match: false,
                    ct_foreign_key: false,
                    ct_table_match: false,
                    tc_foreign_key: false,
                    tc_table_match: false,
                    tt_foreign_key: false,
                }
            ) + {cv_link: false}),
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
