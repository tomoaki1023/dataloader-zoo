# Data-Loader-Zoo

[日本語版 README はこちら](README_JP.md)

## Information
Data-Loader-Zoo is a framework-independent collection of data loaders for deep learning projects. It supports various datasets and provides Python-based implementations (using data formats like numpy arrays and lists) that can be used with different deep learning frameworks. We aim to develop simple data loaders using only Python, without relying on framework-specific data loaders like those in PyTorch or TensorFlow. As a result, our loaders may not be as optimized for performance as the framework-specific ones.

## Background
In deep learning projects, datasets need to be input into neural networks in an appropriate format. However, there are several challenges:

- The effort required to create data loaders for each dataset
- Adapting to different data formats for various deep learning frameworks

Data-Loader-Zoo aims to solve these challenges and improve the efficiency of researchers and developers.

## Features

- Framework-independent: Usable with major deep learning frameworks
- Support for various datasets: From common to specialized datasets (currently limited, but expanding)
- Easy to use: Implemented entirely in Python
- Customizable: Can be extended or modified as needed

## Supported Datasets
Currently, we support the following datasets. While the list is small, we plan to gradually increase the number of supported datasets:

- MNIST dataset
- COCO dataset

## Environment
This project is developed on Ubuntu 22.04.4 LTS using Python 3.10.12. Please install the necessary Python packages as required.

## Usage
Please refer to the `sample.py` file in each dataset directory for examples of how to use the data loaders. The batch data obtained from the data loaders is typically in numpy format. Label information may sometimes be in list format. Therefore, when inputting data into PyTorch models, you may need to convert the numpy data obtained from the data loader to torch.Tensor.

## Contributing
We welcome contributions to the project, such as adding new data loaders or improving existing ones. Here's how you can contribute:

1. Fork this repository
2. Create a new branch (e.g., `git checkout -b feature/add-mnist-loader`)
3. Commit your changes (e.g., `git commit -m 'Add MNIST data loader'`)
4. Push to the branch (e.g., `git push origin feature/add-mnist-loader`)
5. Create a Pull Request

Examples of branch names:
- `feature/add-cifar10-loader`: When adding a loader for the CIFAR-10 dataset
- `bugfix/fix-imagenet-loader`: When fixing a bug in the ImageNet loader
- `docs/update-readme`: When updating the README file

We encourage those who are not familiar with Git to create pull requests as well. Don't worry about making mistakes; let's learn Git together!

We especially welcome beginners and those interested in deep learning. You can contribute in various ways:

- Improving or translating documentation
- Adding support for new datasets
- Reporting bugs or requesting features
- Suggesting code optimizations or improvements

If you have any questions, please don't hesitate to open an Issue. Our community values learning and growth, and we welcome all questions. Let's learn and grow together!

Contributing to this project is an excellent opportunity to develop practical skills in deep learning. Experienced members will support you, so feel free to take on the challenge. Your contributions will help develop this project and the entire deep learning community.

## References
  - https://yann.lecun.com/exdb/mnist
  - https://cocodataset.org