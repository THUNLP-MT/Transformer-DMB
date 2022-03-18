# Dynamic Multi-Branch Layers for On-Device Neural Machine Translation

This is the codes for our work [Dynamic Multi-Branch Layers for On-Device Neural Machine Translation](https://ieeexplore.ieee.org/document/9729651). The implementation is on top of the open-source NMT toolkit [THUMT](https://github.com/THUNLP-MT/THUMT).

## Contents

* [Prerequisites](#prerequisites)
* [Training](#training)
* [Decoding](#decoding)
* [Profile](#profile)
* [License](#license)
* [Citation](#citation)

## Prerequisites

* Python >= 3.7
* tensorflow-cpu >= 2.0
* torch >= 1.7
* torchprofile

Please read the document of [THUMT](https://github.com/THUNLP-MT/THUMT/blob/master/docs/index.md) before using this Repository.

## Training

Training a Transformer-DMB model is nearly the same as a Transformer model, except for the following additional hyperparameters:

* `n`: the number of branches for each layer
* `shared_private`: enable shared-private reparameterization

```
export PYTHONPATH=<path/to/thumt>:$PYTHONPATH

python thumt/bin/trainer.py \
  --input <path/to/src> <path/to/tgt> \
  --model transformer_dmb \
  --vocabulary <path/to/src-vocab> <path/to/tgt-vocab> \
  --parameters=hidden_size=128,filter_size=512,device_list=[0,1,2,3],\
               update_cycle=4,train_steps=100000,\
               shared_source_target_embedding=true,\
               shared_embedding_and_softmax_weights=true,n=4,\
               shared_private=true \
               --hparam_set base
```

## Decoding

The following command decodes an input file with a pre-trained checkpoint.
```
export PYTHONPATH=<path/to/thumt>:$PYTHONPATH

python thumt/bin/translator.py \
  --input <path/to/input> \
  --model transformer_dmb \
  --vocabulary <path/to/src-vocab> <path/to/tgt-vocab> \
  --parameters=device_list=[0,1,2,3] \
  --checkpoint <path/to/checkpoint> \
  --output <path/to/output-file>
```

## Profile

Use the following command to count the number of MultAdds of a tiny Transformer-DMB model with 8 branches.
```
export PYTHONPATH=<path/to/thumt>:$PYTHONPATH

python thumt/bin/trainer.py \
  --input <dummy/src-file> <dummy/tgt-file> \
  --model transformer_dmb \
  --vocabulary <path/to/src-vocab> <path/to/tgt-vocab> \
  --parameters=hidden_size=128,filter_size=512,n=8 \
  --hparam_set base --profile --output /tmp
```

## License

Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes.

## Citation

```
@article{tan2022dynamic,
  author={Tan, Zhixing and Yang, Zeyuan and Zhang, Meng and Liu, Qun and Sun, Maosong and Liu, Yang},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={Dynamic Multi-Branch Layers for On-Device Neural Machine Translation},
  year={2022},
  volume={30},
  pages={958--967},
  doi={10.1109/TASLP.2022.3153257}}

```
