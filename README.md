## Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval

by  Rui Xu, Zongyan Han, Le Hui, Jianjun Qian, and Jin Xie.

### Usage

1. requires:

   ```
   CUDA10 + Pytorch 1.2.0 + Python3
   ```

2. Train:

   ```
   CUDA_VISIBLE_DEVICES=0 python main.py --network DD_GAN_V1 --model_dir DD_GAN_V1  --batch_size 256 --max_epoch 300  --snapshot 50 --phase train
   ```

3. Test:

   ```
   CUDA_VISIBLE_DEVICES=0 python main.py --network DD_GAN_V1 --model_dir DD_GAN_V1  --batch_size 20  --pretrain_model best_full_P.pth --phase test
   ```



### Citation

If you find the code useful, please consider citing:

```
@article{xu2022domain,
  title={Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval},
  author={Xu, Rui and Han, Zongyan and Hui, Le and Qian, Jianjun and Xie, Jin},
  journal={arXiv preprint arXiv:2202.11948},
  year={2022}
}
```

### Acknowledgement

Our word embedding model is from [GloVe](https://github.com/stanfordnlp/GloVe)

Our evaluation code is from [Deep Correlated Holistic Metric Learning for Sketch-based 3D Shape Retrieval](https://github.com/csjinxie/Sketch-based-3D-shape-retrieval)
