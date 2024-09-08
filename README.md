# Volumetric Conditioning Module to Control Pretrained Diffusion Models for 3D Medical Images
Anonymous WACV Applications Track submission Paper ID 1551 for reproducibility of Volumetric Conditioning Module (VCM)

![versatile_VCM_examples](./examples/versatile_VCM.png)
<img src="./examples/LV-spatialcontrol.gif" alt="LV Guidance Viz" width="400"/>
<img src="./examples/Multimodal_spatialcontrol.gif" alt="Multimodal Guidance Viz" width="400"/>
![VCM_details](./examples/VCM_details.png)

## 1. installation

### a. environment setup
```
conda create -n vcm python=3.12 -y
conda activate vcm
python setup.py install
```

### b. accelerate setup

run `accelerate config` command in your shell for accelerate configuration
_The following is an example_

```
----------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine                                                                                                                                                                                                            
----------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?                                                                                                                                                                                          
multi-GPU                                                                                                                                                                                                               
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                                                                              
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes                                                                                      
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                                                                                       
Do you want to use DeepSpeed? [yes/NO]: NO                                                                                                                                                                              
Do you want to use FullyShardedDataParallel? [yes/NO]: NO                                                                                                                                                               
Do you want to use Megatron-LM ? [yes/NO]: NO                                                                                                                                                                           
How many GPU(s) should be used for distributed training? [1]:4                                                                                                                                                          
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:'0,1,2,3'                                                                                                              
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16                                                                                                                                                                                                                    
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

## 2. donwload weights

By using `models/large_files.yml`, donwload the weights for BrainLDM and VCM.
locate them in the `models` directory


## 3. train VCM

You can train your VCM by modifying `acceler-VCM-newSemantics.py` and `newSemantics_loader.py` with your own conditions.

run the train code command
```CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch --num_processes 4 --multi_gpu --gpu_ids='all' --main_process_port 29500 acceler-VCM-newSemantics.py```


## 4. train VCM

You can perform sampling through the condition located in the `data/I-demo` folder via `VCM_sampling.ipynb`. 
Also, you can load and perform sampling of the VCM learned in another condition by referring to the code.