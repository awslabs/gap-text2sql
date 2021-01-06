local configs = [
  # a. base (rerun since align_mat=false was broken, without fix_primary_keys)
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: false,
    merge_types: false,
    fix_primary_keys: false,
  },
  # b. base (rerun since align_mat=false was broken, with fix_primary_keys)
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: false,
    merge_types: false,
    fix_primary_keys: true,
  },
  # c. base + alignment matrix (now with fix_primary_keys)
  {
    align_mat:   true,
    align_loss:  false,
    schema_link: false,
    merge_types: false,
    fix_primary_keys: true,
  },
  # d. base + alignment matrix + alignment loss (now with fix_primary_keys)
  {
    align_mat:   true,
    align_loss:  true,
    schema_link: false,
    merge_types: false,
    fix_primary_keys: true,
  },
  # e. base + schema linking (rerun since align_mat=false was broken, without fix_primary_keys)
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: true,
    merge_types: false,
    fix_primary_keys: false,
  },
  # f. base + schema linking + alignment matrix (now with fix_primary_keys)
  {
    align_mat:   true,
    align_loss:  false,
    schema_link: true,
    merge_types: false,
    fix_primary_keys: true,
  },
  # g. base + schema linking + alignment matrix + alignment loss (now with
  # fix_primary_keys))
  {
    align_mat:   true,
    align_loss:  true,
    schema_link: true,
    merge_types: false,
    fix_primary_keys: true,
  },
  # h. base + merge_types (rerun since align_mat=false was broken, without fix_primary_keys)
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: false,
    merge_types: true,
    fix_primary_keys: false,
  },
  # i. base + merge_types (rerun since align_mat=false was broken, with fix_primary_keys)
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: false,
    merge_types: true,
    fix_primary_keys: true,
  },
  # j. base + alignment matrix + merge_types (now with fix_primary_keys)
  {
    align_mat:   true,
    align_loss:  false,
    schema_link: false,
    merge_types: true,
    fix_primary_keys: true,
  },
  # k. base + alignment matrix + alignment loss + merge_types (now with fix_primary_keys)
  {
    align_mat:   true,
    align_loss:  true,
    schema_link: false,
    merge_types: true,
    fix_primary_keys: true,
  },
];

# Return a string that can be surrouonded with double quotes (") and given as an
# argument for --config-args, as interpreted by Bash.
# Since the string is surrounded by double quotes, variable interpolation will
# occur due to Bash.
# Assumes that no newlines occur inside args.
local manifest_args = function(args)
    std.escapeStringJson(
      std.strReplace(
        std.manifestJsonEx(args + {data_path: '$$PT_DATA_DIR/'}, ''),
        "\n", ""));

local config_gen = function(atts) {
    description: "Ablations for model submitted to Spider on 2019-09-24, att numbers %s" % [atts],

    target: {
      vc: "msrlabs",
      cluster: "rr2",
    },
      
    environment: {
      image: "pytorch/pytorch:0.4.1-cuda9-cudnn7-devel",
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
            name:
            "0924_am%(align_mat)s_al%(align_loss)s_sl%(schema_link)s_fpk%(fix_primary_keys)s_mt%(merge_types)s_att%(att)d" % config_att,
            sku: "G1",
            command: [
              ("philly/entry_point.sh " +
               "python3 train.py " +
               "--config configs/spider-20190205/nl2code-0924-ablations.jsonnet " +
               "--config-args %s " +
               "--logdir \"$$PT_OUTPUT_DIR/logdirs\"") % manifest_args(config_att)
            ],
        }
        for config in configs
        for att in atts
    ]
};

{
    'spider-0924-ablations-att0.yaml': config_gen([0]),
    'spider-0924-ablations-att12.yaml': config_gen([1, 2]),
    'spider-0924-ablations-att3to9.yaml': config_gen([3, 4, 5, 6, 7, 8, 9]),
}
