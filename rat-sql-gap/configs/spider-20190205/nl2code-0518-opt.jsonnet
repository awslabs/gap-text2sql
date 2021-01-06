local _0428_base = import 'nl2code-0428-base.libsonnet';

local decays = {
    linear: {
          name: 'warmup_polynomial',
          power: 1,
    },
    cosine: {
          name: 'warmup_cosine',
          power:: null,
    },
};

function(args) _0428_base(output_from=false) + {
    local lr_s = '%0.1e' % args.lr,
    local wd_s = if args.wd == 0 then '0e0' else '%0.1e' % args.wd,

    model_name: 'lr=%(lr)s,wd=%(wd)s,decay=%(decay)s,att=%(att)d' % (args + {
        lr: lr_s,
        wd: wd_s,
    }),

    model+: {
        encoder+: {
            batch_encs_update: false,
        },
    },

    train+: {
        batch_size: 50,
        max_steps: 20000,

        keep_every_n: 500,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },


    optimizer: {
        name: 'adamw',
        lr: 0.0,
        weight_decay: args.wd,
    },

    lr_scheduler+: decays[args.decay] + {
        start_lr: args.lr,
    },
}
