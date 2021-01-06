'''
python experiments/spider-20190205/eval_20191204_glove_ablations.py \
    --philly-yaml /mnt/wuphillyblob_share/spider/spider-1204-ablations-att0/philly.yaml \
    --blob-root /mnt/wuphillyblob_blob \
    --output /mnt/wuphillyblob_share/spider/spider-1204-ablations-att0/results \
    > eval_20191204_ablations.txt

python experiments/spider-20190205/eval_20191204_glove_ablations.py \
    --philly-yaml /mnt/wuphillyblob_share/spider/spider-1204-ablations-att1to4/philly.yaml \
    --blob-root /mnt/wuphillyblob_blob \
    --output /mnt/wuphillyblob_share/spider/spider-1204-ablations-att1to4/results \
    >> eval_20191204_ablations.txt
'''
import codecs
import json
import re

import enumerate_eval_philly

def create_exp_info(name, commands, data_dir):
    assert len(commands) == 1
    m = re.match(r'.*--config-args "(.*)" --logdir', commands[0])
    assert m is not None
    # Replace \" with ", etc
    args_str = codecs.decode(codecs.encode(m.group(1), 'ascii', 'backslashreplace'), 'unicode-escape')
    args = json.loads(args_str)

    info_dict = {
        'Align matrix': args['align_mat'],
        'Align loss': args['align_loss'],
        'Schema linking': args['schema_link'],
        'Schema edges': args['schema_edges'],
        'Att #': args['att'],
    }
    return info_dict, args_str.replace('$$PT_DATA_DIR', data_dir)

if __name__ == '__main__':
    enumerate_eval_philly.main(create_exp_info, 'configs/spider-20190205/nl2code-1204-glove-ablations.jsonnet')

