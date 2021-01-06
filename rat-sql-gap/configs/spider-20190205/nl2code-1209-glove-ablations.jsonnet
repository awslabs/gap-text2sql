# Remove schema edges: foreign key, primary key, which column is in which table
# -> schema_edges=False
#    **already done in 1204**
# Remove schema linking: n-gram matching between columns and tables
# -> schema_link=False
#    **need to rerun from 1204**
# Remove alignment loss: keeps the matrix between question words and columns, but removes the objective
# -> align_mat=False
#    **already done in 1204**
# Remove alignment loss + alignment matrix
# -> align_mat=False
# -> align_loss=False
#    **already done in 1204**
# Remove everything above (but keep the Transformer)
#    **need to rerun from 1204**

local _1204_glove_ablations = import 'nl2code-1204-glove-ablations.jsonnet';

function(args) _1204_glove_ablations(args) + {
    local preproc_path = args.data_path + 'nl2code-1209-glove-ablations,sc_link=%s' % args.schema_link,

    model+: {
        encoder_preproc+: {
            save_path: preproc_path,
            compute_sc_link: args.schema_link,
        },
        decoder_preproc+: {
            save_path: preproc_path,
        },
    },
}
