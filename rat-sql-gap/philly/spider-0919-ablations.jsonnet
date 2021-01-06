# For alignment, there are two things: the alignment matrix and the alignment
# loss. The alignment matrix by itself could be useful without the alignment
# loss (experimentally, we found this wasn’t true, but I mean to show in the
# paper). So maybe for alignment we should have an ablation where we remove the
# loss but keep the matrix, and another where we remove both.

local configs = [
  # a. base
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: false,
  },
  # b. base + alignment matrix
  {
    align_mat:   true,
    align_loss:  false,
    schema_link: false,
  },
  # c. base + alignment matrix + alignment loss
  {
    align_mat:   true,
    align_loss:  true,
    schema_link: false,
  },
  # d. base + schema linking
  {
    align_mat:   false,
    align_loss:  false,
    schema_link: true,
  },
  # e. base + schema linking + alignment matrix
  {
    align_mat:   true,
    align_loss:  false,
    schema_link: true,
  },
  # f. base + schema linking + alignment matrix + alignment loss
  {
    align_mat:   true,
    align_loss:  true,
    schema_link: true,
  },
];

local config_gen = function(atts) {
    description: "Ablations for model submitted to Spider on 2019-09-19, att numbers %s" % [atts],

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
        name: "am%(align_mat)s_al%(align_loss)s_sl%(schema_link)s_att%(att)d" % (config + {att: att}),
        sku: "G1",
        command: [
          "philly/spider-0919-ablations.sh %(align_mat)s %(align_loss)s %(schema_link)s %(att)d" % (config + {att: att})
        ],
      }
      for config in configs
      for att in atts
    ]
};

{
    'spider-0919-ablations-att0.yaml': config_gen([0]),
    'spider-0919-ablations-att12.yaml': config_gen([1, 2]),
    'spider-0919-ablations-att3to9.yaml': config_gen([3, 4, 5, 6, 7, 8, 9]),
}
