import time
import torch
print(f"from torch.utils.data import DataLoader")
from torch.utils.data import DataLoader
print(f"from utils import *")
from utils import *
print(f"from network.Network import *")
from network.Network import *

from utils.load_train_setting import *
from utils.helper import get_time_ttl_and_eta

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--epoch_number", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])

    a = parser.parse_args()
    g = a.gpu_ids
    d = torch.device(f"cuda:{g[0]}") if torch.cuda.is_available() and g else torch.device("cpu")
    a.device = d
    return a

args = parse_args()
device = args.device

if H != args.H:
    print(f"Change from json to args H: {H} -> {args.H}")
    H = args.H
if W != args.W:
    print(f"Change from json to args W: {W} -> {args.W}")
    W = args.W
if message_length != args.message_length:
    print(f"Change from json to args message_length: {message_length} -> {args.message_length}")
    message_length = args.message_length
if epoch_number != args.epoch_number:
    print(f"Change from json to args epoch_number: {epoch_number} -> {args.epoch_number}")
    epoch_number = args.epoch_number
if batch_size != args.batch_size:
    print(f"Change from json to args batch_size: {batch_size} -> {args.batch_size}")
    batch_size = args.batch_size

print(f"cwd            : {os.getcwd()}")
print(f"pid            : {os.getpid()}")
print(f"host           : {os.uname().nodename}")
print(f"gpu_ids        : {args.gpu_ids}")
print(f"device         : {device}")
print(f"batch_size     : {batch_size}")
print(f"with_diffusion : {with_diffusion}")
print(f"network H      : {H}")
print(f"network W      : {W}")
print(f"network lr     : {lr}")
print(f"message_length : {message_length}")
print(f"only_decoder   : {only_decoder}")
print(f"epoch_number   : {epoch_number}")
print(f"dataset_path   : {dataset_path}")
network = Network(H, W, message_length, noise_layers, device, batch_size,
                  lr, with_diffusion, only_decoder, gpu_ids=args.gpu_ids)

train_dataset = MBRSDataset(dataset_path, H, W, data_file_layout='sub_dirs', return_ori_img=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

print(f"train_continue : {train_continue}")
print(f"result_folder  : {result_folder}")
if train_continue:
    EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
    D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
    network.load_model(EC_path, D_path)

training_start_time = time.time()
b_cnt = len(train_dataloader)
batch_total = epoch_number * b_cnt
batch_iter = 0
for epoch in range(1, epoch_number+1):
    epoch += train_continue_epoch if train_continue else 0
    running_result_sum = {}
    cur_epoch_start_time = time.time()
    for b_idx, images, in enumerate(train_dataloader):
        batch_iter += 1
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        if only_decoder:
            result = network.train_only_decoder(image, message)
        else:
            result = network.train(image, message)

        if len(running_result_sum) == 0:
            running_result_sum = result
        else:
            for key in result:
                running_result_sum[key] += float(result[key])
            # for
        if b_idx % 5 == 0 or b_idx+1 == b_cnt:
            err = running_result_sum["error_rate"] / (b_idx + 1)
            psnr = running_result_sum["psnr"] / (b_idx + 1)
            ssim = running_result_sum["ssim"] / (b_idx + 1)
            elp, eta = get_time_ttl_and_eta(training_start_time, batch_iter, batch_total)
            print(f"E{epoch:03d} B{b_idx:03d}/{b_cnt}: error_rate:{err:.8f}, psnr:{psnr:8.4f}, ssim:{ssim:7.4f}."
                  f" elp:{elp}, eta:{eta}")
        # if
    # for batch
    content = "Epoch " + str(epoch) + " : " + str(int(time.time() - cur_epoch_start_time)) + "\n"
    for key in running_result_sum:
        content += key + "=" + str(running_result_sum[key] / b_cnt) + ","
    content += "\n"

    with open(result_folder + "/train_log.txt", "a") as file:
        file.write(content)
    print(content)

    if epoch % 5 == 0 or epoch == epoch_number:
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + f"EC_{epoch:03d}.pth"
        path_discriminator = path_model + f"D_{epoch:03d}.pth"
        print(f"save_model: {path_encoder_decoder}")
        print(f"save_model: {path_discriminator}")
        network.save_model(path_encoder_decoder, path_discriminator)
    # if
# for epoch

