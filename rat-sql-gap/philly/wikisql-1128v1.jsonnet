# Return a string that can be surrouonded with double quotes (") and given as an
# argument for --config-args, as interpreted by Bash.
# Since the string is surrounded by double quotes, variable interpolation will
# occur due to Bash.
# Assumes that no newlines occur inside args.
local manifest_args = function(args)
    std.escapeStringJson(
      std.strReplace(
        std.manifestJsonEx(args + {data_path: '$$PT_DATA_DIR/wikisql/'}, ''),
        "\n", ""));

local configs = [{}];

local config_gen = function(atts) {
    description: "WikiSQl 2019-11-28 v1, att numbers %s" % [atts],

    target: {
      vc: "msrlabs",
      cluster: "wu2",
    },
      
    environment: {
      image: "pytorch/pytorch:1.3-cuda10.1-cudnn7-devel",
    },

    code: {
      local_dir: "$CONFIG_DIR/../",
    },

    data: {
      local_dir: "$CONFIG_DIR/../../data",
      remote_dir: "spider/bawang",
    },

    jobs: [
        local config_att = config + {att: att};
        {
            name: "wikisql_1128v1_att%(att)d" % config_att,
            sku: "G1",
            command: [
              ("philly/entry_point.sh " +
               "python3 -u train.py " +
               "--config configs/wikisql/nl2code-1128v1.jsonnet " +
               "--config-args %s " +
               "--logdir \"$$PT_OUTPUT_DIR/\"") % manifest_args(config_att)
            ],
        }
        for config in configs
        for att in atts
    ]
};

{
    'wikisql-1128v1.yaml': config_gen([0, 1, 2, 3, 4]),
}
