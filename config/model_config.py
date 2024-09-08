
class defaultCFG():
    def __init__(self) -> None:
        self.BZ = 16
        self.n_epochs = 10000
        self.total_iter = 106 * self.n_epochs 
        self.init_lr = 5e-5
        self.lr_scale = 1/10
        
        self.scale_factor = 0.8962649106979370
        
    def get_training_CFG(self):
        model_CFG = self.get_MCM_3d_CFG()
        train_CFG = {
            'BZ': self.BZ,
            'epoch': self.n_epochs,
            'total_iter': self.total_iter,
            'init_lr': self.init_lr,
            'lr_scale': self.lr_scale,
            'scale_factor': self.scale_factor
        }
        
        return {'model_CFG':model_CFG,
                'train_CFG':train_CFG}
        
    def print_train_CFG(self):
        train_CFG = self.get_training_CFG()['train_CFG']
        for key, value in train_CFG.items():
            print(f'\t\t{key}: {value}')
            
        
        
    def get_AE_CFG(self):
        # the pretrained autoencoder CFG
        return {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 3,
                "num_channels": [
                    64,
                    128,
                    128,
                    128
                ],
                "num_res_blocks": 2,
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "attention_levels": [
                    False,
                    False,
                    False,
                    False
                ],
                "with_encoder_nonlocal_attn": False,
                "with_decoder_nonlocal_attn": False
            }

    def get_DM_CFG(self):
        # the pre-trained diffusion model 
        return  {
                    "spatial_dims": 3,
                    "in_channels": 7,
                    "out_channels": 3,
                    "num_channels": [
                        256,
                        512,
                        768
                        ],
                        "num_res_blocks": 2,
                        "attention_levels": [
                            False,
                            True,
                            True
                        ],
                        "norm_num_groups": 32,
                        "norm_eps": 1e-06,
                        "resblock_updown": True,
                        "num_head_channels": [
                            0,
                            512,
                            768
                        ],
                        "with_conditioning": True,
                        "transformer_num_layers": 1,
                        "cross_attention_dim": 4,
                        "upcast_attention": True,
                        "use_flash_attention": True
                    }
        
    def get_VCM_enc_CFG(self):
        enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 8,
                "num_channels": [
                    16,
                    32,
                    48
                ],
                "num_res_blocks": (2,2,2),
                "attention_levels": [
                    False,
                    False,
                    False
                ],
                "norm_num_groups": 16,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0,
                    0
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
            }  
        
        VCM_enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 3+3+48,
                "out_channels": 64,
                "num_channels": [
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    False,
                    True
                ],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": True
            }
        return VCM_enc_CFG, enc_CFG
