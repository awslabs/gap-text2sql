# In all ablations, we disabled most relation types.

local configs = [
  # a. base
  {
    align_mat:   false,
    align_loss:  false,
  },
  # b. base + alignment matrix
  {
    align_mat:   true,
    align_loss:  false,
  },
  # c. base + alignment matrix + alignment loss
  {
    align_mat:   true,
    align_loss:  true,
  },
];

local config_gen = function(atts) {
    description: "Ablations for model submitted to Spider on 2019-09-21, att numbers %s" % [atts],

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
      {
        name: "norels_am%(align_mat)s_al%(align_loss)s_att%(att)d" % (config + {att: att}),
        sku: "G1",
        command: [
          "philly/spider-0923-ablations.sh %(align_mat)s %(align_loss)s %(att)d" % (config + {att: att})
        ],
      }
      for config in configs
      for att in atts
    ]
};

{
    'spider-0923-ablations-att0.yaml': config_gen([0]),
    'spider-0923-ablations-att12.yaml': config_gen([1, 2]),
    'spider-0923-ablations-att3to9.yaml': config_gen([3, 4, 5, 6, 7, 8, 9]),
}
