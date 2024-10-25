# TPNet

> Text-prompt Camouflaged Instance Segmentation with Graduated Camouflage Learning,            
> *ACM MM 2024*

## Abstract
Camouflaged instance segmentation (CIS) aims to seamlessly detect and segment objects blending with their surroundings. While existing CIS methods rely heavily on fully-supervised training with massive precisely annotated data, consuming considerable annotation efforts yet struggling to segment highly camouflaged objects accurately. Despite their visual similarity to the background, camouflaged objects differ semantically. Since text associated with images offers explicit semantic cues to underscore this difference, in this paper we propose a novel approach: the first \textbf{T}ext-\textbf{P}rompt based weakly-supervised camouflaged instance segmentation method named TPNet, leveraging semantic distinctions for effective segmentation. Specifically, TPNet operates in two stages: initiating with the generation of pseudo masks followed by a self-training process. In the pseudo mask generation stage, we innovatively align text prompts with images using a pre-training language-image model to obtain region proposals containing camouflaged instances and specific text prompt. Additionally, a Semantic-Spatial Iterative Fusion module is ingeniously designed to assimilate spatial information with semantic insights, iteratively refining pseudo mask. In the following stage, we employ Graduated Camouflage Learning, a straightforward self-training optimization strategy that evaluates camouflage levels to sequence training from simple to complex images, facilitating for an effective learning gradient. Through the collaboration of the dual phases, our method offers a comprehensive experiment on two common benchmark and demonstrates a significant advancement, delivering a novel solution that bridges the gap between weak-supervised and high camouflaged instance segmentation.


## Usage
### Install
```bash
conda create --name tpnet python=3.8 -y
conda activate tpnet
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
pip install -r requirements.txt

```
- [DETReg](https://github.com/amirbar/DETReg/): Follow the instructions at DETReg for installation.
- [OSFormer](https://github.com/PJLallen/OSFormer): Follow the instructions at OSFormer to obtain the dataset and generate the dataset format.
- [CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt): Download CLIP pre-trained ViT-B/16.
### Directory
The directory should be like this:

````
-- TPNet
-- data (train dataset and test dataset)
-- model (saved model)
-- pre (pretrained model)
-- data (train dataset and test dataset)
   |-- CIS
   |   |-- Train_Image_CAM
   |   |-- COD10K
   |   |-- NC4K
   ...
   
````

### Train
#### stage1
```bash
cd maskcut
python maskcut.py --vit-arch base --patch-size 8 --tau 0.15 --fixed_size 480 --N 3 --dataset-path path/to/Train/data --out-dir output/path
```
* We adopt pre-trained DINO and CLIP as pretrain model.

#### stage2
```bash
cd tpnet
python train_net.py --num-gpus 4 --config-file /media/data2/HZT/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml 
```

## Acknowledgement
We borrowed the code from [CutLER](https://github.com/facebookresearch/CutLER/tree/main) and [pytorch_grad_camg](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a/). Thanks for their wonderful works.
```
## Citation
@inproceedings{xia2024text,
  title={Text-prompt Camouflaged Instance Segmentation with Graduated Camouflage Learning},
  author={Xia, Changqun and Qiao, Shengye and Li, Jia and others},
  booktitle={ACM Multimedia 2024},
  year={2024}
}
```
