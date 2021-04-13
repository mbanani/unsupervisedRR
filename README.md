UnsupervisedR&R: Unsupervised Pointcloud Registration via Differentiable Rendering
====================================

This repository holds all the code and data for our recent work on unsupervised point cloud
registration:

**[UnsupervisedR&R: Unsupervised Pointcloud Registration via Differentiable Rendering][1]**  
[Mohamed El Banani](https://mbanani.github.io), [Luya Gao](https://scholar.google.com/citations?user=OHk0dbgAAAAJ), [Justin Johnson](https://web.eecs.umich.edu/~justincj/)

If you find this code useful, please consider citing:  
```text
@inProceedings{elbanani2021unsupervisedrr,
  title={{UnsupervisedR&R: Unsupervised Pointcloud Registration via Differentiable Rendering}},
  author={El Banani, Mohamed and Gao, Luya and Johnson, Justin},
  booktitle={CVPR},
  year={2021},
}
```

If you have any questions about the paper or the code, please feel free to email me at
mbanani@umich.edu 


Usage Instructions
------------------

1. [How to setup your environment?][2]
2. [How to download and setup the datasets?][3]
3. [How to train models?][4]
4. [How to run inference with pretrained checkpoints?][5]

Acknowledgments
---------------
We would like to thank the reviewers and area chairs for their valuable comments and suggestions. 
We also thank Nilesh Kulkarni, Karan Desai, Richard Higgins, and Max Smith for many helpful
discussions and feedback on early drafts of this work. 

We would also like to acknowledge the following repositories and users for making great code openly
available for us to use:

- [@pytorch/pytorch](https://www.github.com/pytorch/pytorch)
- [@facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) for a great 3D vision
  library. 
- [@intel-isl/Open3D](https://github.com/intel-isl/Open3D) for easy to use implementations of
  traditional methods and great visualizers. 
- [@chrischoy](https://github.com/chrischoy/DeepGlobalRegistration) and
  [@zgojcic](https://github.com/zgojcic/3D_multiview_reg) for releasing excellent code that allows
for easy geometric registration. 


[1]: https://mbanani.github.io/unsupervisedRR/
[2]: https://github.com/mbanani/unsupervisedRR/tree/main/docs/environment.md 
[3]: https://github.com/mbanani/unsupervisedRR/tree/main/docs/datasets.md 
[4]: https://github.com/mbanani/unsupervisedRR/tree/main/docs/training.md
[5]: https://github.com/mbanani/unsupervisedRR/tree/main/docs/evaluation.md 
