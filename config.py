rotation_dim = 6
bone_length_dim = 1
context_length = 20
gen_frames = 5

z_dim = gen_frames * rotation_dim * 7
hid_lyrs = [32, 32, 32, 32, 32]
head_num = 16
tgt_all_lyr = True
dropout = 0.1

epochs = 1000
pos_weight = 0.025

early_stopping_patience = 20
early_stopping_min_delta = 0.01

checkpoint_interval = 50

logs_dir = "./logs"
checkpoints_dir = "./checkpoints"
