# Pretrained, fixed word embeddings.
# - emb: glove-42B, bpemb-10k, or bpemb-100k
# - min_freq: 3, 50
# - output_from

local _0220 = import 'nl2code-0220.jsonnet';
local PREFIX = 'data/spider-20190205/';

local embedders = {
    'glove-42B': {
        name: 'glove',
        kind: '42B',
    },
    'bpemb-10k': {
        name: 'bpemb',
        dim: 300,
        vocab_size: 10000,
    },
    'bpemb-100k': {
        name: 'bpemb',
        dim: 300,
        vocab_size: 100000,
    },
};

function(args) _0220({
    output_from: false, 
    qenc: 'eb',
    ctenc: 'ebs',
    upd_steps: 4,
    max_steps: 40000,
    batch_size: 10
}) + {
    model_name: 'emb=%(emb)s,min_freq=%(min_freq)d,enc_size=%(enc_size)d,dec_size=%(dec_size)d,att=%(att)d' % args,
    model+: {
        encoder+: {
            word_emb_size: 300,
            recurrent_size: args.enc_size,
        },
        decoder+: {
            desc_attn: 'mha-1h',
            rule_emb_size: args.dec_size / 2,
            node_embed_size: args.dec_size / 4,
            enc_recurrent_size: args.enc_size,
            recurrent_size: args.dec_size,
        },

        encoder_preproc+: {
            word_emb: embedders[args.emb],
            # Set to false for fixed word embeddings.
            count_tokens_in_word_emb_for_vocab: false,
            min_freq: args.min_freq,

            save_path: PREFIX + 'nl2code-0420-sketch,emb=%(emb)s,min_freq=%(min_freq)d/' % args,
        },
        decoder_preproc+: {
            grammar+: {
                include_columns: false,
            },
            word_emb:: null,
            count_tokens_in_word_emb_for_vocab:: null,
        },
    },
}
