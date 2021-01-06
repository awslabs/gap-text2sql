function(args) {
    local _fixed_set = std.set(args.fixed),
    // local _check = [(assert std.setMember(elem, ['data', 'init', 'model']); null) for elem in _fixed_set],

    local lr_s = '%0.1e' % args.lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,att=%(att)d' % (args + {
        lr: lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        encoder+: {
            batch_encs_update: false,
        },
    },

    train+: {
        batch_size: args.bs,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },

    lr_scheduler+: {
        start_lr: args.lr,
        end_lr: args.end_lr,
    },
}
