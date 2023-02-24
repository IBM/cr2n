# Neural-based classification rule learning for sequential data

Implementation of the **Convolutional Rule Neural Network (CR2N)** presented in "_Neural-based classification rule learning for sequential data_" ICLR 2023 paper [[OpenReview](https://openreview.net/forum?id=7tJyBmu9iCj), [arXiv](https://arxiv.org/abs/2302.11286)].


Notebooks are provided to replicate the experiments.

- main_peptides.ipynb on the UCI Anticancer dataset.
- main_synth.ipynb on synthetic datasets available [here](https://github.com/IBM/synth-sequential-datasets).


Complementary:

- main_synth_downsampling.ipynb - quick experiment on unbalanced dataset processed with downsampling (following reviewer's comment).


## Requirements

Dependencies listed in `requirements.txt`.
Python 3.8 was used for the experiments.

For the synthetic datasets: [https://github.com/IBM/synth-sequential-datasets](https://github.com/IBM/synth-sequential-datasets).

## Cite
Collery, M., Bonnard, P., Fages, F., & Kusters, R. (2023). Neural-based classification rule learning for sequential data. International Conference on Learning Representations.

```
@inproceedings{collery2023neural,
  title={Neural-based classification rule learning for sequential data},
  author={Collery, Marine and Bonnard, Philippe and Fages, Fran{\c{c}}ois and Kusters, Remy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
## License
Apache 2.0, detailed LICENSE is available [here](https://github.com/IBM/cr2n/blob/main/LICENSE).