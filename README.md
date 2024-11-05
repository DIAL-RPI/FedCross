# FedCross
Federated Cross Learning for Medical Image Segmentation

This is a python (PyTorch) implementation of **Federated Cross Learning (FedCross)** method proposed in our paper [**"Federated Cross Learning for Medical Image Segmentation"**](https://openreview.net/forum?id=DrZbwobH_zo) published in *Medical Imaging with Deep Learning 2023* conference (Nashville, Tennessee, United States, Jul. 10-12, 2023). A preprint version of this paper is also available on [arXiv](https://arxiv.org/abs/2204.02450).

## Citation
  *X. Xu, H. H. Deng, T. Chen, T. Kuang, J. C. Barber, D. Kim, J. Gateno, J. J. Xia, and P. Yan, "Federated Cross Learning for Medical Image Segmentation," in Medical Imaging with Deep Learning 2023. Nashville, Tennessee, United States, Jul. 10-12, 2023.*

    @inproceedings{Xu2023FedCross, 
      title={Federated Cross Learning for Medical Image Segmentation},
      author={Xuanang Xu and Hannah H. Deng and Tianyi Chen and Tianshu Kuang and Joshua C. Barber and Daeseung Kim and Jaime Gateno and James J. Xia and Pingkun Yan},
      booktitle={Medical Imaging with Deep Learning},
      year={2023},
      url={https://openreview.net/forum?id=DrZbwobH_zo}
    }

## Update
  - **Nov 21, 2023**: Fix a bug (`read_image` function was missing in `utils.py` file).

## Abstract
Federated learning (FL) can collaboratively train deep learning models using isolated patient data owned by different hospitals for various clinical applications, including medical image segmentation. However, a major problem of FL is its performance degradation when dealing with data that are not independently and identically distributed (non-iid), which is often the case in medical images. In this paper, we first conduct a theoretical analysis on the FL algorithm to reveal the problem of model aggregation during training on non-iid data. With the insights gained through the analysis, we propose a simple yet effective method, federated cross learning (FedCross), to tackle this challenging problem. Unlike the conventional FL methods that combine multiple individually trained local models on a server node, our FedCross sequentially trains the global model across different clients in a round-robin manner, and thus the entire training procedure does not involve any model aggregation steps. To further improve its performance to be comparable with the centralized learning method, we combine the FedCross with an ensemble learning mechanism to compose a federated cross ensemble learning (FedCrossEns) method. Finally, we conduct extensive experiments using a set of public datasets. The experimental results show that the proposed FedCross training strategy outperforms the mainstream FL methods on non-iid data. In addition to improving the segmentation performance, our FedCrossEns can further provide a quantitative estimation of the model uncertainty, demonstrating the effectiveness and clinical significance of our designs. Source code is publicly available at [https://github.com/DIAL-RPI/FedCross](https://github.com/DIAL-RPI/FedCross).

## An illustration of non-iid problem in FL model aggregation
The model aggregation process in FL may lead to sub-optimal solution when dealing with non-iid data. In the figure below, (a) and (b) show that the locally trained models $θ^{J+1}_k$ and $θ^{J+1}_m$ each individually achieve their minimal in the loss landscape of the client dataset $D_k$ and $D_m$, respectively. (c) indicates that the aggregated FL model $θ^′$ is located at a non-minimal position in the global loss landscape of $D_k \cup D_m$.
<img src="./fig1.png"/>

## Method
### Training schemes of (a) FedAvg, (b) FedCross, and (c) FedCrossEns
<img src="./fig2.png"/>

## Contact
You are welcome to contact us:  
  - [xux12@rpi.edu](mailto:xux12@rpi.edu) ([Dr. Xuanang Xu](https://superxuang.github.io/))
