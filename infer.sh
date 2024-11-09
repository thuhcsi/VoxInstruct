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

