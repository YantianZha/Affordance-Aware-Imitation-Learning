# Contrastively Learning Visual Attention as Affordance Cues from Demonstrations for Robotic Grasping 


[Paper](https://drive.google.com/file/d/1NjTrwPTFbpvktXgQRivUFIH4GlB4Ksr3/view?usp=sharing) | [Project Website](https://sites.google.com/asu.edu/affordance-aware-imitation/project) 

![plot](./config/mug_exp1.png)

### Cite our Paper
```
@article{zha2021contrastively,
  title={Contrastively Learning Visual Attention as Affordance Cues from Demonstrations for Robotic Grasping},
  author={Zha, Yantian and Bhambri, Siddhant and Guan, Lin},
  journal={The IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021}
}
```

This repository includes codes for reproducing the experiments (Table. 1) in the paper:
1. Ours (Full Model): Siamese + Coupled Triplet Loss
2. Ours (Ablation): Siamese + Normal Triplet Loss [To Add]
3. Ours (Ablation): Without Contrastive Learning [To Add]
4. Baseline [To Add]

### Dependencies (`pip install` with python=3.6 or 3.7):
1. pybullet
2. PyTorch
3. ray

### Dataset and Mug Models:
Please download [here](https://drive.google.com/drive/folders/11Tde7DxHVYrnt43tzGM6uyxSDaKRh4NY?usp=sharing). Change the path correspondingly in the file "src/grasp_bc_13_a.json"

### Training:
1. Modify ```trainFolderDir```,```testFolderDir```, and ```obj_folder``` in `src/grasp_bc_13_a.json`; Modify ```result_path``` and ```model_path``` in all training scripts (e.g. `trainGrasp_full.py`).
2. Train the full model with coupled triplet loss by running ```python3 trainGrasp_full.py src/grasp_bc_13_a```; If you train the model on a multi-GPU server, you could run ```CUDA_VISIBLE_DEVICES=1 nohup python3 trainGrasp_full.py src/grasp_bc_13_a > nohup.out &``` and monitor the training by the command ```tail -f nohup.out```.
3. To visualize grasping rollout, you can go to the ```result_path``` that you set at step 1; you can easily check results by training epoches or mug IDs.

### FAQ
1. Where is the implementation of coupled triplet loss?
   
   Please check [this line](https://github.com/YantianZha/Affordance-Aware-Imitation-Learning/blob/b2a48077970f75bfbab98d31d10afd425a962581/trainGrasp_full.py#L322).
2. What are the mugs used and how they are different from each other? 
   
   Please check [this line]().   
   
3. Why the visualization images in trajectory folders are different in brightnesses?

### Acknowledgement
The code is based on [this repo](https://github.com/irom-lab/PAC-Imitation).