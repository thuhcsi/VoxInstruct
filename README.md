
# VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling

This is the official PyTorch implementation of **"[VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling](https://dl.acm.org/doi/10.1145/3664647.3681680)", accepted by ACM MM2024 as Oral.

Demopage is available at: [VoxInstruct Demo](https://voxinstruct.github.io/VoxInstruct/) 

In this repository, we provide the VoxInstruct model, inference scripts, and the checkpoint that has been trained on the internal large-scale <instruction, speech> dataset. 


## Installation

You can install the necessary dependencies using the `requirements.txt` file with Python 3.9.18:

```bash
conda create -n voxinstruct python=3.9.18
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Due to storage limitations, we have saved the model checkpoints on Google Drive at https://drive.google.com/drive/folders/1LoVAnMiwAq4X-OI0U2UQJsIEs7G6o0L0?usp=drive_link. 

You can download the models and save them in the `pretrained` folder.


## Inference

To run inference, use the following command in `infer.sh` and input your instructions at the `examples/example_instructions.txt`.


```bash
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
    --ar_config ./configs/train_ar.yaml \
    --nar_config ./configs/train_nar.yaml \
    --ar_ckpt ./pretrained/voxinstruct-sft-checkpoint/ar_1800k.pyt \
    --nar_ckpt ./pretrained/voxinstruct-sft-checkpoint/nar_1800k.pyt \
    --synth_file  ./examples/example_instructions.txt \
    --out_dir ./results  \
    --device cuda \
    --vocoder vocos \
    --cfg_st_on_text 1.5 \
    --cfg_at_on_text 3.0 \
    --cfg_at_on_st 1.5 \
    --nar_iter_steps 8
```
You can try other cfg value for better performance.


## License

The code and weights in this repository is released under the MIT license as found in the [LICENSE](LICENSE) file.


## Citation
Please cite our paper if you find this work useful:
```bibtex
@inproceedings{zhou2024voxinstruct,
  title={VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling},
  author={Zhou, Yixuan and Qin, Xiaoyu and Jin, Zeyu and Zhou, Shuoyi and Lei, Shun and Zhou, Songtao and Wu, Zhiyong and Jia, Jia},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={554--563},
  year={2024}
}
```




