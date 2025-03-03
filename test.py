import datetime
import network.noise_layers.crop as crop
import torchvision.transforms.functional as ttf
import torchvision.utils as tvu

def image_quality_evalute(images, encode_images):
    """

    evaluate the visual quality

    """

    # calculate psnr and ssim

    psnr = -kornia.losses.psnr_loss(encode_images.detach(), images, max_val=2.).item()

    ssim = 1 - 2 * kornia.losses.ssim_loss(encode_images.detach(), images, max_val=1., window_size=5,
                                           reduction="mean").item()

    return psnr, ssim

def parse_args():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default="normal_test")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
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
EC_path = args.EC_path
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

def test_by_crop(crop_rate, test_scenario):
    global batch_iter
    random_crop = crop.RandomCrop(crop_rate)
    print(f"crop: {crop_rate} {test_scenario} =================================================")
    test_result_sum = {"acc_rate": 0.0, "msg_loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    test_result_cnt = 0

    network.encoder_decoder.eval()
    network.discriminator.eval()
    print(f"crop:{crop_rate:.2f} {test_scenario} start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-6])
    for b_idx, (sml_images, big_images) in enumerate(test_dataloader):
        batch_iter += 1
        # small images: sml_images, such as 128*128
        # batch size must be 1 !!!
        sml_images, big_images = sml_images.to(network.device), big_images.to(network.device)
        message = torch.Tensor(np.random.choice([0, 1], (sml_images.shape[0], message_length)))
        messages = message.to(network.device)
        with torch.no_grad():
            # use device to compute
            sml_encoded_images = network.encoder_decoder.module.encoder(sml_images, messages)
            sml_encoded_images = sml_images + (sml_encoded_images - sml_images) * strength_factor

            b, c, h, w = big_images.shape
            if b != 1:
                raise ValueError(f"In testing, batch size ({b}) must be 1.")
            if test_scenario == 's2b_residual':
                sml_residual = sml_encoded_images - sml_images
                big_residual = ttf.resize(sml_residual, (h, w))
                big_encoded_images = big_images + big_residual
            elif test_scenario == 's2b_direct':
                big_encoded_images = ttf.resize(sml_encoded_images, (h, w))
            else:
                raise ValueError(f"Invalid test_scenario: {test_scenario}")
            if args.todo == 'sample_save':
                save_testing_images(big_images, big_encoded_images, test_scenario, b_idx, b)

            cropped_big_encoded_images = random_crop(big_encoded_images)
            cropped_sml_encoded_images = ttf.resize(cropped_big_encoded_images, (128, 128))

            decoded_messages = network.encoder_decoder.module.decoder(cropped_sml_encoded_images)

            psnr, ssim = image_quality_evalute(big_images, big_encoded_images)
        # with
        error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
        msg_loss = network.criterion_MSE(decoded_messages, messages)

        result = {"acc_rate": 1 - error_rate, "msg_loss": msg_loss, "psnr": psnr, "ssim": ssim}

        test_result_cnt += 1
        for key in result:
            test_result_sum[key] += float(result[key])

        if b_idx % 10 == 0 or b_idx+1 == b_cnt:
            acc_rate   = test_result_sum["acc_rate"] / test_result_cnt
            msg_loss   = test_result_sum["msg_loss"] / test_result_cnt
            psnr       = test_result_sum["psnr"] / test_result_cnt
            ssim       = test_result_sum["ssim"] / test_result_cnt
            elp, eta = get_time_ttl_and_eta(start_time, batch_iter, batch_total)
            msg = (f"crop:{crop_rate:.2f} {test_scenario} B{b_idx:03d}/{b_cnt}: acc_rate:{acc_rate:.8f}, "
                   f"msg_loss:{msg_loss:.8f}, psnr:{psnr:10.6f}, ssim:{ssim:.6f}. elp:{elp}, eta:{eta}")
            print(msg)
        # if
    # for
    print(f"crop:{crop_rate:.2f} {test_scenario} ended:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-6])
    print(f"crop:{crop_rate:.2f} {test_scenario} Average")
    for key in test_result_sum:
        print("  ", key, "=", str(test_result_sum[key] / test_result_cnt))
    print("")   # empty lines in log. to separate different crop rate.
    print("")

def save_testing_images(big_images, big_encoded_images, test_scenario, b_idx, b_sz):
    dir_name = "./images_ori_vs_enc/"
    os.makedirs(dir_name, exist_ok=True)
    train_label = parse_training_label()
    stem = f"{test_scenario}_{train_label}"
    for i_idx, (img_ori, img_enc) in enumerate(zip(big_images, big_encoded_images)):
        if b_sz == 1:
            f_path = os.path.join(dir_name, f"{stem}_image{b_idx:02d}.png")
        else:
            f_path = os.path.join(dir_name, f"{stem}_batch{b_idx:02d}_image{i_idx:02d}.png")
        img = torch.cat((img_ori, img_enc), dim=2) # [c, h, w]
        print(f"img: {img.shape}")
        img = (img + 1.) / 2.
        img = torch.clamp(img, 0., 1.)
        tvu.save_image(img, f_path)
        print(f"Saved: {f_path}")
    # for

def parse_training_label():
    """
    from EC_path, parse label string, such as "TrainRandomCrop0.1"
    EC_path="../output1_train_test_128x128_msgLen30_TrainRandomCrop0.1/results/MBRS_m30/models/EC_010.pth"
    """
    global EC_path
    ec = EC_path
    ec = ec.replace("\\", "/")
    arr =ec.split("/")
    label_pre = "TrainRandomCrop"
    idx = -1
    dir_segment = None
    for tmp in arr:
        idx = tmp.find(label_pre)
        if idx >= 0:
            dir_segment = tmp
            break
        # if
    # for
    if idx < 0:
        return ""
    label_len = len(label_pre) + 3
    label = dir_segment[idx:idx+label_len]
    return label

def list_dataset_files(dataset_path):
    file_list = os.listdir(dataset_path)
    file_list.sort()
    print(f"dataset_path: {dataset_path}")
    print(f"file count: {len(file_list)}...")
    for idx, f_name in enumerate(file_list):
        print(f"  {idx:d}: {f_name}")
    print(f"file count: {len(file_list)}...Done")

if args.todo == 'normal_test':
    crop_list = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91,
                 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.04, 0.01]
    # crop_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.04, 0.01]
elif args.todo == 'sample_save':
    crop_list = [1]
    list_dataset_files(args.dataset_path)
else:
    raise ValueError(f"Invalid todo: {args.todo}")
test_scenario_arr = ['s2b_residual', 's2b_direct']
start_time = time.time()
b_cnt = len(test_dataloader)
batch_total = len(crop_list) * len(test_scenario_arr) * b_cnt
batch_iter = 0

for s in test_scenario_arr:
    for c in crop_list:
        test_by_crop(c, s)
    print(f"")
    print(f"------------------------------------------------------------")
    print(f"------------------------------------------------------------")
    print(f"------------------------------------------------------------")
    print(f"")
    # for
# for
