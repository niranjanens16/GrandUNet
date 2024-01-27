
CSE 575: statistical machine learning - Niranjan E N S

Research project aimed to develop an unified architecture combining the strengths of 
Graph U-nets and Graph Neural diffusion (GRAND). This innovative approach utilizes 
Graph U-Net’s pooling, unpooling and residual connections combined with diffusion principles
to address oversmoothing in Graph Neural Networks (GNN), and make use of
different graph structures for having a higher resolution view of the network
structure. The resulting architecture combines the power of convolutional-like
processing with continuous diffusion steps, offering a promising solution for nodelevel
prediction tasks in graph data. Our experiments with 3 different datasets
demonstrate comparable results with the state-of-the art architectures.

virtual env
conda create –name grandUNet python=3.8

packages
pip install ogb pykeops                                                                                                                                       
pip install torch                                                                                                                                   
pip install torchdiffeq                                                                                                                          
pip install torch-scatter                                                                       
pip install torch-sparse                                                                      
pip install torch-cluster                                                                     
pip install torch-spline-conv                                                                  
pip install torch-geometric
Pip install ray

To run:
python grandUNet.py --dataset {graph/node dataset} --segment {provide segmentation for distributed training}



