# DualViT: A Hierarchical Vision Transformer for Broad and Fine Class Embeddings

[![Paper](https://img.shields.io/badge/Paper-Springer-blue.svg)](https://doi.org/10.1007/978-3-031-78166-7_3)
[![Conference](https://img.shields.io/badge/Conference-ICPR%202024-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-78166-7_3)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **DualViT**, presented at the **International Conference on Pattern Recognition (ICPR) 2024**.

DualViT is a hierarchical vision transformer that simultaneously learns broad-class and fine-class embeddings through dual transformer encoders trained alternately, encoding hierarchical knowledge via tensor products between coarse and fine class representations.

---

## Highlights

- Dual encoder architecture learning hierarchical class embeddings (broad + fine)
- Tensor product fusion of coarse and fine representations
- Achieves competitive accuracy with fewer training epochs
- Evaluated on CIFAR-10, CIFAR-100, and ImageNet-1k

---

## Paper

> **DualViT: A Hierarchical Vision Transformer for Broad and Fine Class Embeddings**
> Ankita Chatterjee, Sandip Dutta, Jayanta Mukhopadhyay, Partha Pratim Das
> *International Conference on Pattern Recognition (ICPR), 2024*
> DOI: [10.1007/978-3-031-78166-7_3](https://doi.org/10.1007/978-3-031-78166-7_3)

---

## Installation

```bash
git clone https://github.com/Dutta-SD/DualViT.git
cd DualViT
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

**CIFAR-10:**
```bash
python cifar10.py
```

**CIFAR-100:**
```bash
python cifar100.py
```

**ImageNet-1k:**
```bash
python imagenet1k.py
```

---

## Project Structure

```
DualViT/
├── dualvit/
│   ├── model/          # DualViT architecture
│   ├── lightning/      # PyTorch Lightning training modules
│   ├── constants.py    # Hyperparameters and config
│   └── factory.py      # Model factory
├── analysis/           # Notebooks and experiments
├── cifar10.py          # CIFAR-10 training script
├── cifar100.py         # CIFAR-100 training script
└── imagenet1k.py       # ImageNet-1k training script
```

---

## Citation

```bibtex
@inproceedings{chatterjee2024dualvit,
  title={DualViT: A Hierarchical Vision Transformer for Broad and Fine Class Embeddings},
  author={Chatterjee, Ankita and Dutta, Sandip and Mukhopadhyay, Jayanta and Das, Partha Pratim},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2024},
  publisher={Springer},
  doi={10.1007/978-3-031-78166-7_3}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
