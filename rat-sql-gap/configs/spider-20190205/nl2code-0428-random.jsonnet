local _0428_base = import 'nl2code-0428-base.libsonnet';

function(args) _0428_base(output_from=false) + {
    local _fixed_set = std.set(args.fixed),
    // local _check = [(assert std.setMember(elem, ['data', 'init', 'model']); null) for elem in _fixed_set],

    model_name: 'fixed=%(fixed)s,att=%(att)d' % (args + {
        fixed: std.join('+', _fixed_set)
    }),

    train+: {
        model_seed: if std.setMember('model', _fixed_set) then 1234 else args.att,
        data_seed:  if std.setMember('data', _fixed_set) then 1234 else args.att,
        init_seed:  if std.setMember('init', _fixed_set) then 1234 else args.att,
    }
}
