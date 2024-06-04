import torch
import os
from PIL import Image, ImageDraw, ImageFont
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from dataset.clp2k import ImageDataset, load_data
from train_SLDM_clp import AutoencoderKL, UNetModel, UniPCMultistepScheduler, SDMLDMPipeline


data_root = "./data/clp_q8k"
resolution = 512
segmap_channels = 15
num_classes = 15
num_inference_steps=20
s=1.5
batch_size=1
output_dir = "./results/re3_20000_50steps_drawing"
unet = "./sldm-vae-15-clp_re3/checkpoint-20000/unet_ema"
learning_rate = 1e-4
global_step = 0

def test(vae, unet, noise_scheduler, accelerator, weight_dtype, data_ld, 
                   resolution=512, g_step=2, save_dir=output_dir):
    scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)
    pipeline = SDMLDMPipeline(
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        resolution=resolution,
        resolution_type="clp"
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.enable_xformers_memory_efficient_attention()

    rgb_label_dir = "./data/clp_q8k/drawing_rgb_labels"
    generator = None

    #for i, batch in enumerate(data_ld):
    for i, (image_tensor, out_dict, file_name) in enumerate(data_ld):
        #if i > 2:
        #   break
        images = []
        #segmap_name = os.path.basename(batch[1]['file_name'][0])
        # get arr_image and out_dict
        #arr_image, out_dict, _ = batch
        # get label
        label = out_dict['label']
        segmap_name = file_name
        segmap_name = segmap_name[0]
        file_name_without_extension, _ = os.path.splitext(segmap_name)
        correct_file_name = file_name_without_extension + ".png"
        rgb_segmap_path = os.path.join(rgb_label_dir, correct_file_name)
        rgb_segmap = Image.open(rgb_segmap_path)

        with torch.autocast("cuda"):
            segmap = preprocess_input(label, num_classes=num_classes)
            segmap = segmap.to("cuda").to(torch.float16)
           
            image = pipeline(segmap=segmap[0][None,:], generator=generator, batch_size=batch_size,
                              num_inference_steps=num_inference_steps, s=s).images
            images.extend(image)
            save_single_images(images, segmap_name)
            
            
        merge_images_with_rgb(images, rgb_segmap, segmap_name)

def save_single_images(images, segmap_name):
    for k, image in enumerate(images):
        filename = f"{segmap_name}"
        #path = os.path.join(accelerator.logging_dir, f"step_{step}", "singles", filename)
        path = os.path.join(output_dir, "singles", filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        image.save(path)

def merge_images_with_rgb(images, rgb_segmap, segmap_name):
    total_width = sum(img.width for img in images) + rgb_segmap.width
    max_height = max(max(img.height for img in images), rgb_segmap.height)
    combined_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    combined_image.paste(rgb_segmap, (x_offset, 0))
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.load_default()
    #draw.text((10, 10), segmap_name, (255, 255, 255), font=font)

    merge_filename = f"{segmap_name}"
    merge_path = os.path.join(output_dir, "merges", merge_filename)
    os.makedirs(os.path.split(merge_path)[0], exist_ok=True)
    combined_image.save(merge_path)

def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data = data.to(dtype=torch.int64)

    # create one-hot label map
    label_map = data
    bs, _, h, w = label_map.size()
    input_label = torch.FloatTensor(bs, num_classes, h, w).zero_().to(data.device)
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    return input_semantics

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    #logging_dir="logging",  
    log_with="tensorboard",
)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

vae = AutoencoderKL.from_pretrained("./VAE")
unet = UNetModel.from_pretrained(unet)


val_dataloader, _ = load_data(
    dataset_mode="clp2k",
    data_dir=data_root,
    batch_size=1,
    image_size= resolution,
    is_train=False)

pipeline = SDMLDMPipeline(
    vae=vae,
    unet=unet,
    scheduler=UniPCMultistepScheduler(),
    torch_dtype=torch.float32,
)

test(vae, unet, UniPCMultistepScheduler(), accelerator, weight_dtype, val_dataloader, resolution=resolution, g_step=global_step, save_dir=output_dir)
