# GGP-SPC
**Paper:** Solving Inverse  Computational imaging Problems Using  Deep Generative Gradient of Priors

**Authors:** Ruirui Liu, Jinglong Xing, Haiyu Zheng, Hulin Zhou, Qiurong Yan, Yuhao Wang, Qiegen Liu*



*Data : 11/2020*

*Version : 1.0*

*The code and the algorithm are for non-comercial use  only.*

*Copyright 2020, Department of Electronic Information Engineering, Nanchang University.*

Reconstruction of signals from compressively sensed measurements is an ill-posed inverse problem. Recently, deep generative models have demonstrated great potential in solving inverse imaging problems. In this work, gradients of generative priors (GGP) based on denoising score matching (DSM) are proposed for Single Pixel Camera (SPC) imaging reconstruction. Rather than the existing deep generative models that often estimate the data distribution, the proposed GGP model estimates the gradients of the data distribution.

### Reconstruction 

python3.5 demo_ggp.py --model ggp --exe GGP_EXE --config ggp.yml --doc cifar3ch --test --image_folder result

**Poster at the conference "computational imaging technology and Applications"**

![post](https://github.com/yqx7150/GGP-SPC/blob/main/ggp_spc/Poster.Jpeg)

### Checkpoints

We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/13FQCLDPI8lQ7awAAPrtF5w). key number is "ovid"

### Sampling parameters

We provide Phi of with different rate. You can download the model  from [Baidu Drive](https://pan.baidu.com/s/1QCcOYA8HQrF_yrbMhajZ8Q). key number is "ue4g"
## Other Related Projects
  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2008.06284)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)

  * Progressive Colorization via Interative Generative Models  
[<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)  

 
