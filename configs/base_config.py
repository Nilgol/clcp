"""A configuration template to set volatile model and training parameters conveniently.
Path to a config should be provided as command-line argument ('-cfg') when calling main.

Note that filepaths are not considered volatile and therefor not currently included as config
parameters. Instead, set them once as constants within the respective scripts, namley:
    -train.py
    -data/a2d2_dataset.py
"""

# Main model parameters
embed_dim = 384
temperature = 0.07
batch_size = 256


# Point cloud encoder parameters, stored as a dict
projection_type = "linear"
freeze_point_cloud_encoder_weights = False

point_cloud_encoder_params = {
    "freeze_encoder_weights": freeze_point_cloud_encoder_weights,
    "projection_type": projection_type,
}

# Training parameters
num_workers = 8
learning_rate = 1e-4 # base learning rate after warmup
weight_decay = 1e-5
max_epochs = 100
val_ratio = 0.15
augment = False # whether to use more aggressive image augmentations

# Optimizer parameters, expected as dict
optimizer_params = { # AdamW assumed
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "betas": (0.9, 0.98),
}

# Scheduler parameters, expected as dict
scheduler_params = {
    "warmup": { # linear warmup scheduler
        "start_factor": 1e-6 / learning_rate, # initial learning rate
        "total_iters": 5, # warmup epochs
    },
    "cosine": { # cosine annealing with warm restarts scheduler
        "T_0": 1, # length of initial learning rate cycle in epochs
        "T_mult": 2, # growth factor for cycle length
        "eta_min": 1e-5, # minimum learning rate at the end of cycle
    },
}