from .settings import *
import os
import time

'''
params setting
'''
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
filename = os.path.join(parent_dir, "train_settings.json")
settings = JsonConfig()
settings.load_json_file(filename)

with_diffusion = settings.with_diffusion
only_decoder = settings.only_decoder

project_name = settings.project_name
dataset_path = settings.dataset_path
val_dataset_path = settings.val_dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
save_images_number = settings.save_images_number
lr = settings.lr
H, W, message_length = settings.H, settings.W, settings.message_length,
noise_layers = settings.noise_layers

'''
file preparing
'''
full_project_name = project_name + "_m" + str(message_length)
# for noise in noise_layers:
# 	full_project_name += "_" + noise
result_folder = "results/" + full_project_name + "/"
if not os.path.exists(result_folder): os.makedirs(result_folder)
if not os.path.exists(result_folder + "images/"): os.makedirs(result_folder + "images/")
if not os.path.exists(result_folder + "models/"): os.makedirs(result_folder + "models/")
with open(result_folder + "/train_params.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"

	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"

	print(content)
	file.write(content)
with open(result_folder + "/train_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
with open(result_folder + "/val_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
