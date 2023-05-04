# DECISION
DECISION: Deep rEcurrent Controller for vISual NavigatION

This repo contains the implementation of the controllers presented in the paper \
Deep Visual Naivgation under Partial Observability, ICRA 2022 \
This work follows our previous work Intention-Net, CoRL 2017 [[paper](https://arxiv.org/abs/1710.05627)]

## Contents
More specifically, we implemented
* [Intention-Net](https://arxiv.org/abs/1710.05627) in Torch 1.6.0
* ConvLSTM cells, proposed by the [paper](https://arxiv.org/abs/1506.04214)
* Truncated Backpropagation Through Time (TBPTT), first appeared in the [thesis](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)

## Requirements
* [PyTorch](http://pytorch.org/) (ver. 1.6+ required)
* torchvision 0.7.0, tensorboard 2.4.1 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Refer to ```environmet.yml``` for more details of the environment. 

## Get started
A sample dataset consisting 200 images are available in the repo. To start training our DECISION controller, run the following
```
git clone https://github.com/AdaCompNUS/DECISION.git
cd DECISION
python main.py
```
The default hyperparameters may not be optimal. Refer to the below tips for reproduction. 

## Tips for use
* Do not trust the test loss value. In our case, it is not a good indicator of a control's online performance (i.i.d assumption). Thus, directly evaluate the policy online even for hyperparameter turning. The test value is only a sanity check that the training is ongoing. 
* Important hyperparameters: model capacity, learning rate and decay schedule, dropout, reasampling scheme (see below), and arguments ```--frame-interval```, ```--k1```,  ```--k2-n```. Refer to the papers for our training hyperparameters. 
* One key to leanring a policy that actually works is to adjust the dataset distribution via resampling. Finding the right resampling scheme requires trials and errors. The procedure is dataset-dependent and our implementation is in the ```SeqDataset``` class in ```dataset.py```. 
* Using a small batch size (> 8 samples per GPU) may bring troubles. If you observe the test loss increases while training loss decreases, try using a larger batch size or commenting out ```model.eval()``` before evaluating. If this helps, the problem is the incorrect batch statistics tracked by the norm layers. Solutions: (1) Use a larger batch size. (2) Implement batch norm layers that synchronize statistics across GPUs (```nn.SyncBatchNorm``` might be useful, not tested yet). (3) Use [Group Norm](https://arxiv.org/abs/1803.08494) (```nn.GroupNorm```) for all layers and tune the hyperparam ```num_groups```. 

## Examples
To train the original I-Net, the following command can be run:
`python main.py --dataset-path <path/to/dataset> --num-frames 1 --batch-size 128 --modes 3 --num-modes 3 --model inet --gpu 0,1`

To switch to DECISION, it is necessary to ensure `num_frames > 1` to provide some history for the ConvLSTM to process. A suggestion is to run:
`python main.py --dataset-path <path/to/dataset> --num-frames 35 --batch-size 32 --modes 3 --num-modes 3 --model decision --gpu 0,1`

## Citation
```bibtex
@INPROCEEDINGS{9811598,
  author={Ai, Bo and Gao, Wei and Vinay and Hsu, David},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Deep Visual Navigation under Partial Observability}, 
  year={2022},
  volume={},
  number={},
  pages={9439-9446},
  doi={10.1109/ICRA46639.2022.9811598}
 }
```
