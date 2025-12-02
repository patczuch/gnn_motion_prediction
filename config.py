rotation_dim = 6
position_dim = 3
context_length = 20
gen_frames = 5

z_dim = 144
hid_lyrs = [32, 32, 32, 32, 32]
head_num = 96
tgt_all_lyr = True

epochs = 1000
pos_weight = 0.03

early_stopping_patience = 20
early_stopping_min_delta = 0.01

checkpoint_interval = 50

logs_dir = "./logs"
checkpoints_dir = "./checkpoints"
