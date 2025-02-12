from .settings import *
import os
import time

'''
params setting
'''
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
filename = os.path.join(parent_dir, "test_settings.json")
settings = JsonConfig()
settings.load_json_file(filename)

with_diffusion = settings.with_diffusion

# dataset_path = settings.dataset_path
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
save_images_number = settings.save_images_number
lr = 1e-3
H, W, message_length = settings.H, settings.W, settings.message_length
noise_layers = settings.noise_layers

result_folder = os.path.join(parent_dir, "results", settings.result_folder)
os.makedirs(result_folder, exist_ok=True)
test_base = "test_"
# for layer in settings.noise_layers:
# 	test_base += layer + "_"
test_param = os.path.join(result_folder, f"{test_base}{strength_factor}_params.json")
test_log   = os.path.join(result_folder, f"{test_base}{strength_factor}_log.txt")
with open(test_param, "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)

	with open(filename, "r") as setting_file:
		content = setting_file.read()
		file.write(content)

with open(test_log, "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
