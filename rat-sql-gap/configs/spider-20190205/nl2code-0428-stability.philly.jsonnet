local _0428_base = import 'nl2code-0428-base.libsonnet';
local _0428_stability = import 'nl2code-0428-stability.libsonnet';

function(args) _0428_base(output_from=false, data_path=args.data_path) + _0428_stability(args) + {
    log: {
        reopen_to_flush: true,
    }
}
