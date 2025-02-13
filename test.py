import datetime

def parse_args():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--EC_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)

    a = parser.parse_args()
    g = a.gpu_ids
    d = torch.device(f"cuda:{g[0]}") if torch.cuda.is_available() and g else torch.device("cpu")
    a.device = d
    return a

args = parse_args()
device = args.device
gpu_ids = args.gpu_ids
batch_size = args.batch_size

print(f"from torch.utils.data import DataLoader...")
from torch.utils.data import DataLoader
print(f"from utils import *...")
from utils import *
print(f"from network.Network import *...")
from network.Network import *
print(f"from utils.load_test_setting import *...")
from utils.load_test_setting import *

from utils.helper import get_time_ttl_and_eta

if H != args.H:
    print(f"Change from json to args H: {H} -> {args.H}")
    H = args.H
if W != args.W:
    print(f"Change from json to args W: {W} -> {args.W}")
    W = args.W
if message_length != args.message_length:
    print(f"Change from json to args message_length: {message_length} -> {args.message_length}")
    message_length = args.message_length

# EC_path = "../checkpoints/MBRS_256_256/EC_" + str(model_epoch) + ".pth"
EC_path = args.EC_path or "../output2_train_256x256/results/MBRS_m64__2025_01_26__18_26_06/models/EC_025.pth"
ds_path = args.dataset_path
print(f"cwd            : {os.getcwd()}")
print(f"pid            : {os.getpid()}")
print(f"host           : {os.uname().nodename}")
print(f"gpu_ids        : {gpu_ids}")
print(f"device         : {device}")
print(f"batch_size     : {batch_size}")
print(f"with_diffusion : {with_diffusion}")
print(f"EC_path        : {EC_path}")
print(f"strength_factor: {strength_factor}")
print(f"network lr     : {lr}")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, gpu_ids=gpu_ids)
print(f"network.load_model_ed(EC_path)...")
network.load_model_ed(EC_path)
print(f"network.load_model_ed(EC_path)...Done")

test_dataset = MBRSDataset(ds_path, H, W, transform_type=1, return_ori_img=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print(f"dataset_path   : {ds_path}")
print(f"H              : {H}")
print(f"W              : {W}")
print(f"test_dataset   : {len(test_dataset)}")
print(f"test_dataloader: {len(test_dataloader)}")

test_result_sum = {
    "error_rate": 0.0,
    "msg_loss": 0.0,
    "psnr": 0.0,
    "ssim": 0.0
}

saved_iterations = np.random.choice(np.arange(len(test_dataloader)), size=save_images_number, replace=False)
saved_all = None
print(f"save_images_number: {save_images_number}")
print(f"saved_iterations  : {saved_iterations}")

num = 0
b_cnt = len(test_dataloader)
network.encoder_decoder.eval()
network.discriminator.eval()
start_time = time.time()
print("start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-6])
for b_idx, images in enumerate(test_dataloader):
    image = images.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
    with torch.no_grad():
        # use device to compute
        images, messages = images.to(network.device), message.to(network.device)
        encoded_images = network.encoder_decoder.module.encoder(images, messages)
        encoded_images = images + (encoded_images - image) * strength_factor
        noised_images = network.encoder_decoder.module.noise([encoded_images, images])

        decoded_messages = network.encoder_decoder.module.decoder(noised_images)

        enc_img_detach = encoded_images.detach()
        psnr = kornia.losses.psnr_loss(enc_img_detach, images, 2).item()
        ssim = 1 - 2 * kornia.losses.ssim_loss(enc_img_detach, images, window_size=5, reduction="mean").item()
    # with
    '''
    decoded message error rate
    '''
    error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
    msg_loss = network.criterion_MSE(decoded_messages, messages)

    result = {
        "error_rate": error_rate,
        "msg_loss": msg_loss,
        "psnr": psnr,
        "ssim": ssim,
    }

    num += 1
    for key in result:
        test_result_sum[key] += float(result[key])

    if b_idx in saved_iterations:
        if saved_all is None:
            saved_all = get_random_images(image, encoded_images, noised_images)
        else:
            saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

    if b_idx % 10 == 0 or b_idx+1 == b_cnt:
        error_rate = test_result_sum["error_rate"] / num
        msg_loss   = test_result_sum["msg_loss"] / num
        psnr       = test_result_sum["psnr"] / num
        ssim       = test_result_sum["ssim"] / num
        elp, eta = get_time_ttl_and_eta(start_time, b_idx+1, b_cnt)
        msg = (f"B{b_idx:03d}/{b_cnt}: error_rate:{error_rate:.8f}, msg_loss:{msg_loss:.8f}, "
               f"psnr:{psnr:10.6f}, ssim:{ssim:.6f}. elp:{elp}, eta:{eta}")
        print(msg)
        with open(test_log, "a") as file:
            file.write(msg)
    # if
# for
print("ended:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-6])

'''
test results
'''
content = "Average : \n"
for key in test_result_sum:
    content += key + "=" + str(test_result_sum[key] / num) + ","
content += "\n"

with open(test_log, "a") as file:
    file.write(content)

print(content)
folder = result_folder + "images/"
os.makedirs(folder, exist_ok=True)
print(f"save_images() folder  : {folder}")
filepath = save_images(saved_all, "test", folder, resize_to=(W, H))
print(f"save_images() filepath: {filepath}")

