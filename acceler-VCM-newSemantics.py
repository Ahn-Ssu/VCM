import os
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader
from monai.utils import set_determinism
from tqdm import tqdm

from accelerate import Accelerator

from generative.inferers import LatentDiffusionInferer 
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from model.vcm import VCM
from config.model_config import defaultCFG
from newSemantics_loader import MRI_dataset

from misc import misc
import yaml
from datetime import date
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    
# for reproducibility purposes set a seed
set_determinism(42)
cfg = defaultCFG()

accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])
device = accelerator.device
print(f'\t {accelerator.is_main_process=}, {device=}')

# cmd : CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch --num_processes 4 --multi_gpu --gpu_ids='all' --main_process_port 29500 acceler-VCM-newSemantics.py
misc.setup_for_distributed(accelerator.is_local_main_process) # verbose =1 only for main process

print('dataset prep')
three_modality = ['new']
train_transforms = transforms.Compose([
        transforms.ConcatItemsd(keys=three_modality, name='label', dim=0),
    ]
)
print('\t trasfrom done')

train_dataset = MRI_dataset('/root/data/new_semantics500', transform=train_transforms)

BZ = 2
train_loader = DataLoader(train_dataset, batch_size=BZ, shuffle=True, num_workers=8, persistent_workers=True, drop_last=True)
print('\t datset, BZ, loader done')
print(f'\t BZ = {BZ}')

print('model construction, load the weights')
autoencoder = AutoencoderKL(**cfg.get_AE_CFG())
AE_weight_path = 'weights/autoencoder.pth' # the AE of brainLDM
autoencoder.load_state_dict(torch.load(AE_weight_path, map_location='cpu'))
print(f"\t AE done")

diffusion = DiffusionModelUNet(**cfg.get_DM_CFG())
Diff_weight_path = 'weights/diffusion_model.pth' # brainLDM
diffusion.load_state_dict(torch.load(Diff_weight_path, map_location='cpu'))
print(f"\t Diffusion done")

VCM_enc_CFG, enc_CFG = cfg.get_VCM_enc_CFG()
vcm = VCM(out_dim=3, diff_CFG=VCM_enc_CFG, enc_CFG=enc_CFG)
print(f"\t VCM done")


print('training stratagy - DDPM schduler, scale_factor, optimizer, scheduler')

train_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
train_scheduler = train_scheduler

scale_factor = cfg.scale_factor
inferer = LatentDiffusionInferer(train_scheduler, scale_factor=scale_factor)
print(f"\t DDPmScheduler, scale_factor done")

n_epochs = cfg.n_epochs
init_lr = cfg.init_lr
scale = cfg.lr_scale

target_iter = n_epochs * 62 # 62 is # number of train dataset // the size of BZ

optimizer_vcm = torch.optim.AdamW(params=vcm.parameters(), lr=init_lr)
scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer_vcm, first_cycle_steps=target_iter, max_lr=init_lr, min_lr=init_lr*scale, warmup_steps=target_iter//30, gamma=0.8)
global_steps = 0
# global_steps = 1000 * 62 # run epoch number * iteration per epoch
print(f"\t training CFG done")

if accelerator.is_main_process:
    from torch.utils.tensorboard import SummaryWriter
    out_dir = f'/root/vcm/VCM/out/data500/newSemantics/VCM/{date.today()}'
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)
    d = {'train_CFG':cfg.get_training_CFG(), 'model_CFG':VCM_enc_CFG, 'enc_CFG':enc_CFG}
    with open(f'{out_dir}/cfg.yaml', 'w') as file:
        yaml.dump(d, file)

    print(f'\t saved CFG -- ["model_CFG", "train_CFG"]')
    cfg.print_train_CFG()


val_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
val_scheduler.set_timesteps(num_inference_steps=200)

print(type(vcm))



vcm, train_loader, optimizer_vcm, scheduler = accelerator.prepare(
    vcm, train_loader, optimizer_vcm, scheduler
)
print(type(vcm))

autoencoder = accelerator.prepare_model(model = autoencoder)
diffusion = accelerator.prepare_model(model= diffusion)

# vcm.load_state_dict(torch.load('/root/vcm/VCM/out/data500/newSemantics/VCM/2024-09-02/log/1000/vcm_wegith.pt'))
# scheduler.step(global_steps)

if isinstance(autoencoder, torch.nn.parallel.distributed.DistributedDataParallel):
    autoencoder.module.eval()
    diffusion.module.eval()
    vcm.module.train()
else:
    autoencoder.eval()
    diffusion.eval()
    vcm.train()

for epoch in range(n_epochs):
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=80)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        z_mu, z_sigma, cond, label, path, = batch
        cond = cond.view(BZ, 1, 4)
        
        z_mu = z_mu.contiguous()
        z_sigma = z_sigma.contiguous()
        label = label.contiguous()
        
        
        optimizer_vcm.zero_grad(set_to_none=True)
            
        # 1. extract latent representation via the VAE
        with torch.no_grad():
            z = autoencoder.module.sampling(z_mu, z_sigma) if isinstance(autoencoder, torch.nn.parallel.distributed.DistributedDataParallel) else autoencoder.sampling(z_mu, z_sigma)
            z = z.contiguous() * scale_factor
            
        # 2. Generate random noise (epilson ~ N(0,I))
        epsilon = torch.randn_like(z).to(device)
        
        # 3. Create timestep t
        t = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
        
        # 4. make z_t by adding some noise from t and scheduler 
        z_t = train_scheduler.add_noise(original_samples=z,
                                    noise=epsilon, 
                                    timesteps=t)
        z_t = z_t.contiguous()
        
        # 5. modified the conditions to concat on latent z 
        cond_concat = cond.view(BZ, 4, 1, 1, 1)
        cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(z_t.shape[2:]))

        # 6. get noise prediction from the pretrained diffusion model 
        with torch.no_grad():
            epsilon_t = diffusion(torch.cat((z_t, cond_concat), dim=1),
                                    timesteps=t,
                                    context=cond)
        
        # 7. feed the diffusion priors and novel conditions into VCM networks via concat and get a gamma_t (scale) and a beta_t (shift)
        gamma_t, beta_t = vcm(x=torch.cat([z_t, epsilon_t], dim=1),
                            y=label,
                            timesteps=t)
        
        # 8. scale and shift epsilon_t in element-wise manner 
        epsilon_p_t = epsilon_t * (1 + gamma_t) + beta_t
        
        MSE_loss = F.mse_loss(epsilon_p_t.float(), epsilon.float())
        lambda_x = 1
        L1_loss = torch.norm(gamma_t, p=1) + torch.norm(beta_t, p=1)
        BZ, c, h, w, c = z_t.size()
        lambda_1 = 1 / (BZ * c * h * w * c)
        
        MCM_loss = lambda_x * MSE_loss + lambda_1 * L1_loss
        accelerator.backward(MCM_loss)
        optimizer_vcm.step()
        scheduler.step(global_steps)
        global_steps += 1 
        
        
        epoch_loss += MCM_loss.item()

        if accelerator.is_main_process:
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1), 'Lr': scheduler.get_lr()[0]})
            writer.add_scalar("Epoch Loss", epoch_loss / (step + 1), global_steps)
            writer.add_scalar("Learning Rate", scheduler.get_lr()[0], global_steps)  # Assuming single learning rate
    
    ################### Training Monitor: sampling comparison ####################
    if (epoch) % 500 == 0 and accelerator.is_main_process:
        
        val_path = f'{out_dir}/log/{epoch}'
        os.makedirs(val_path, exist_ok=True)
        torch.save(vcm.state_dict(), f'{val_path}/vcm_weight.pt')
    ################### Training Monitor: sampling comparison ####################
    
if accelerator.is_main_process:
    val_path = f'{out_dir}/log/{n_epochs}'
    os.makedirs(val_path, exist_ok=True)
    torch.save(vcm.state_dict(), f'{val_path}/vcm_weight.pt')