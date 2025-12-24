import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from os.path import join as pjoin

from diffusers import DDIMScheduler
from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from options.denoiser_option import arg_parse

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils import paramUtil

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from data.t2m_unlearning import create_unlearning_dataset, simple_collate, UnlearningDataset
from peft import LoraConfig, get_peft_model


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, opt.kinematic_chain, joint, title=caption, fps=20)


def load_and_freeze_vae(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name, 'opt.txt')
    vae_opt = get_opt(opt_path, opt.device)

    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.to(opt.device)
    print(f'Loading VAE Model {opt.vae_name}')
    return model


if __name__ == '__main__':
    opt = arg_parse(True)
    fixseed(opt.seed)

    # models & noise scheduler
    vae = load_and_freeze_vae(opt)
    denoiser = Denoiser(opt, vae.opt.latent_dim)
    ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    denoiser.load_state_dict(ckpt["denoiser"], strict=False)

    # LoRA injection
    for param in denoiser.parameters():
        param.requires_grad = False

    # Define Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj'],
        lora_dropout=0.1,
        bias="none"
    )

    # Inject
    denoiser = get_peft_model(denoiser, peft_config)
    denoiser.print_trainable_parameters()
    # Explicitly set to training mode to ensure gradient tracking for LoRA modules
    denoiser.train()
    # Expected: ~0.1% to 1% trainable params

    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    num_params = sum(param.numel() for param in denoiser.parameters_without_clip())
    print('Total trainable parameters of all models: {}M'.format(num_params/1_000_000))

    # evaluation setup
    wrapper_opt = get_opt(opt.dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(opt.dataset_opt_path, 32, 'val', device=opt.device)

    # dataset & dataloader
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    # train_split_file = pjoin(opt.data_root, 'train.txt')
    # val_split_file = pjoin(opt.data_root, 'val.txt')

    # train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    # val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)

    
    t2m_base_dir = "./dataset/humanml3d"
    text_dir = os.path.join(t2m_base_dir, "texts")
    motion_dir = os.path.join(t2m_base_dir, "new_joint_vecs")

    NEUTRAL_ID = "000447"

    train_df, val_df, final_df = create_unlearning_dataset(t2m_base_dir, text_dir, motion_dir, NEUTRAL_ID)
    
    train_dataset = UnlearningDataset(
        dataframe=train_df,
        mean_path=os.path.join(t2m_base_dir, "Mean.npy"),
        std_path=os.path.join(t2m_base_dir, "Std.npy")
    )
    val_dataset = UnlearningDataset(
        dataframe=val_df,
        mean_path=os.path.join(t2m_base_dir, "Mean.npy"),
        std_path=os.path.join(t2m_base_dir, "Std.npy")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=simple_collate,
        num_workers=2,
        drop_last=True
    )

    # train
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=plot_t2m)

    # Define the directory for saving the LoRA model
    lora_save_dir = os.path.join(opt.checkpoints_dir, "t2m", "lora_finetuned", "model")
    os.makedirs(lora_save_dir, exist_ok=True)

    # Define the full path for the LoRA model checkpoint
    lora_model_path = os.path.join(lora_save_dir, "lora_finetuned.tar")

    # Save the state_dict of the LoRA model
    torch.save(denoiser.state_dict(), lora_model_path)

    print(f"✅ LoRA fine-tuned model saved to: {lora_model_path}")