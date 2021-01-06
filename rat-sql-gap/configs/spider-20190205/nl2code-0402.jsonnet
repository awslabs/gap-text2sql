# Pretrained word embeddings passed through learned linear layers.
# Question, column, and table encoders each have separate linear layers.
# We also set the number of multi-head attention heads to 1.
# - emb: glove-42B, bpemb-10k, or bpemb-100k
# - min_freq: 3, 50
# - output_from

local _0401 = import 'nl2code-0401.jsonnet';
function(args) _0401(args) + {
    model+: {
        encoder+: {
            question_encoder: ['emb', 'linear', 'bilstm'],
            column_encoder: ['emb', 'linear', 'bilstm-summarize'],
            table_encoder: ['emb', 'linear', 'bilstm-summarize'],
        },
    },
}
