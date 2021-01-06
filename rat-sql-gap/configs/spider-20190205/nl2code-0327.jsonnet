local _0220 = import 'nl2code-0220.jsonnet';

function(args) _0220({
    output_from: args.output_from, 
    qenc: 'eb',
    ctenc: 'ebs',
    upd_steps: 4,
    max_steps: 40000,
    batch_size: 10
}) + {
    model_name: 'rerun,output_from=%(output_from)s,att=%(att)d' % args
}
