# Training Data Attribution in Diffusion Models via Mirrored Unlearning and Noise-Consistent Skew

_This is the repository for the MUCS paper. Although it includes the code we used to perform the experiments reported in the paper, this repository is intended just for illustrative purposes. Please do not expect direct execution to work on the first try._

### Abstract

Training data attribution (TDA) should enable generative model interpretability and foster a variety of related downstream tasks. Nonetheless, current TDA approaches lack reliability and robustness, preventing their adoption in real-world setups. In this paper, we take a decisive step towards more reliable and robust TDA for diffusion models. We propose to perform TDA with mirrored unlearning and noise-consistent skew (MUCS). The idea is to fine-tune a second model with bounded mirrored gradient ascent, and to measure the normalized skew of this model with respect to the original one using consistent noise samples. We show that, while being conceptually simple and generic, MUCS systematically outperforms existing methods on three different datasets by a large margin. We additionally study the effect that core design choices have on final performance, and analyze novel aspects regarding the overlap of influential instances across generated items and the potential of ensembling TDA approaches. We believe that our findings may have broader implications for more general unlearning setups, as well as for tasks requiring the comparison of diffusion losses. 

### Authors

Joan Serrà, Dipam Goswami, Fabio Morreale, Wei-Hsiang Liao, & Yuki Mitsufuji.

### Reference

J. Serrà, D. Goswami, F. Morreale, W.-H. Liao, & Y. Mitsufuji (2026). Training Data Attribution in Diffusion Models via Mirrored Unlearning and Noise-Consistent Skew. [ArXiv: 2605.17938](https://arxiv.org/abs/2605.17938).


## Preparation

### Environment

MUCS requires python>=3.10. We used python 3.10.12.

You should be able to create the environment by running [install_requirements.sh](install_requirements.sh). However, we recommend to just check inside that file and do it step by step.

### Additional folder structure

Create a `pointer_to/` folder and then symlinks to cache/data/logs storage:

```
pointer_to/cache -> /real/path/to/cache/
pointer_to/data -> /real/path/to/preproc_data/
pointer_to/logs -> /real/path/to/logs/
pointer_to/plots -> /real/path/to/plots/
```
```bash
...
ln -s /real/path/to/plots pointer_to/plots
```

### Preprocessing script

CIFAR10:

```bash
OMP_NUM_THREADS=1 python scripts/preproc_data/cifar10.py --tmp_dir=pointer_to/cache/ --dest_dir=pointer_to/data/cifar10
```

ArtBench10:
```bash
OMP_NUM_THREADS=1 python scripts/preproc_data/artbench10.py --source_dir=/group2/ds/data/artbench10/ --dest_dir=pointer_to/data/artbench10/
```

COCO: 

```bash
OMP_NUM_THREADS=1 python scripts/preproc_data/coco.py --tmp_dir=pointer_to/cache/ --dest_dir=pointer_to/data/coco/
```

## Full pipeline

To run the full pipeline, do:
```bash
python pipeline.py --what=tpre,gpre,a,tpost,gpost --job=cifar10-o_dit_edm --data_seed=1 --attrib=random
```
where:
* `tpre` means training of the first model
* `gpre` means generation for the first model
* `a` means compute attribution
* `tpost` means training of the second model
* `gpost` means generation for the second model

To run the eval script:
```bash
python eval.py --jobs=cifar10_dit_edm_ds1,cifar10_dit_edm_ds2 --methods=random,mucs
```

We use slurm to run our scripts, so expect to see some slurm boilerplate code.

## License

The code in this repository is released under the Apache-2.0 license as found in the [LICENSE file](LICENSE).

## Notes

* If using this code or parts of it in any way, please cite the reference above.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.
