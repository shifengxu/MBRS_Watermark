import time

print(f"from torch.utils.data import DataLoader")
from torch.utils.data import DataLoader
print(f"from utils import *")
from utils import *
print(f"from network.Network import *")
from network.Network import *

from utils.load_train_setting import *
from utils.helper import get_time_ttl_and_eta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"cwd            : {os.getcwd()}")
print(f"pid            : {os.getpid()}")
print(f"host           : {os.uname().nodename}")
print(f"device         : {device}")
print(f"batch_size     : {batch_size}")
print(f"with_diffusion : {with_diffusion}")
print(f"network H      : {H}")
print(f"network W      : {W}")
print(f"network lr     : {lr}")
print(f"message_length : {message_length}")
print(f"only_decoder   : {only_decoder}")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)

print(f"dataset_path     : {dataset_path}")
print(f"val_dataset_path : {val_dataset_path}")
train_dataset = MBRSDataset(dataset_path, H, W, transform_type=2, data_file_layout='sub_dirs')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataset = MBRSDataset(val_dataset_path, H, W, transform_type=1)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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
for epoch in range(epoch_number):
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
            print(f"E{epoch:03d} B{b_idx:03d}/{b_cnt}: error_rate:{err:.4f}, psnr:{psnr:8.4f}, ssim:{ssim:7.4f}."
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

    if epoch % 5 == 0 or epoch == epoch_number - 1:
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + f"EC_{epoch:03d}.pth"
        path_discriminator = path_model + f"D_{epoch:03d}.pth"
        print(f"save_model: {path_encoder_decoder}")
        print(f"save_model: {path_discriminator}")
        network.save_model(path_encoder_decoder, path_discriminator)
    # if
# for epoch

def make_validation():
    val_result = {
        "error_rate": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "g_loss": 0.0,
        "g_loss_on_discriminator": 0.0,
        "g_loss_on_encoder": 0.0,
        "g_loss_on_decoder": 0.0,
        "d_cover_loss": 0.0,
        "d_encoded_loss": 0.0
    }


    saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
    saved_all = None

    start_time = time.time()
    val_b_cnt = len(val_dataloader)
    for i, images in enumerate(val_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

        for key in result:
            val_result[key] += float(result[key])

        if i in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, encoded_images, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)
        # if
    # for

    save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

    '''
    validation results
    '''
    content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
    for key in val_result:
        content += key + "=" + str(val_result[key] / val_b_cnt) + ","
    content += "\n"

    with open(result_folder + "/val_log.txt", "a") as file:
        file.write(content)
    print(content)

