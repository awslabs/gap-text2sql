local baseline = {
    schema_edges: true,
    schema_link: true,
    align_mat: true,
    align_loss: true,
};

local configs = [
  #baseline,

  # Remove schema edges: foreign key, primary key, which column is in which table
  # -> schema_edges=false
  #baseline + {schema_edges: false},

  # Remove schema linking: n-gram matching between columns and tables
  # -> schema_link=false
  baseline + {schema_link: false},

  # Remove alignment loss: keeps the matrix between question words and columns, but removes the objective
  # -> align_loss=false
  #baseline + {align_loss: false},

  # Remove alignment loss + alignment matrix
  # -> align_mat=False
  # -> align_loss=False
  #baseline + {align_loss: false, align_mat: false},

  # Remove everything above (but keep the Transformer)
  {
    schema_edges: false,
    schema_link: false,
    align_mat: false,
    align_loss: false,
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
    description: "Ablations for Spider glove from 1209, att numbers %s" % [atts],

    target: {
      vc: "msrlabs",
      cluster: "rr2",
    },
      
    environment: {
      image: "pytorch/pytorch:nightly-devel-cuda9.2-cudnn7",
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
            "1204_se%(schema_edges)s_sl%(schema_link)s_am%(align_mat)s_al%(align_loss)s_att%(att)d" % config_att,
            sku: "G1",
            command: [
              ("philly/entry_point_cuda92.sh " +
               "python3 -u train.py " +
               "--config configs/spider-20190205/nl2code-1209-glove-ablations.jsonnet " +
               "--config-args %s " +
               "--logdir \"$$PT_OUTPUT_DIR/logdirs\"") % manifest_args(config_att)
            ],
        }
        for config in configs
        for att in atts
    ]
};

{
    'spider-1209-ablations-att0to4.yaml': config_gen([0, 1, 2, 3, 4]),
}
