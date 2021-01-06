# Pretrained word embeddings passed through learned linear layers.
# Question, column, and table encoders each have separate linear layers.
# We also set the number of multi-head attention heads to 1.
# - emb: glove-42B, bpemb-10k, or bpemb-100k
# - min_freq: 3, 50
# - output_from

local _0402 = import 'nl2code-0402.jsonnet';
function(args) _0402(args) + {
    model+: {
        decoder+: {
            desc_attn: 'sep'
        },
    },
}
