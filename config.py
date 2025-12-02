rotation_dim = 6
position_dim = 3
context_length = 20
gen_frames = 5

z_dim = 72
hid_lyrs = [24, 24, 24]
head_num = 32
tgt_all_lyr = True

epochs = 1000
pos_weight = 0.03

early_stopping_patience = 20
early_stopping_min_delta = 0.01

checkpoint_interval = 50

logs_dir = "./logs"
checkpoints_dir = "./checkpoints"
