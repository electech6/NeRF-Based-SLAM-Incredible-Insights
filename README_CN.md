# NeRF-Based-SLAM-Incredible-Insights
## 项目简介
欢迎来到 **“NeRF-Based-SLAM-Incredible-Insights”** 仓库。 该项目旨在提供对各种基于 NeRF（神经辐射场）的 Slam（同时定位和建图）算法的全面见解。 如果您对基于 NeRF 的 Slam 算法充满热情并希望深入研究其功能和代码库，那么您来对地方了。

如果您发现此存储库有用，请考虑 **引用 和 加星** 此列表。 请随意与其他人分享这个项目！





## 目录

该仓库包含：

1. **详细文档**介绍各种NeRF-Based Slam算法，阐明其基本原理和算法工作流程，见解和注解等内容详见[Paper Insights]和[Code Notes]和[Tracking Insights]模块。
2. 对于选中的NeRF-Based Slam算法，给出**代码注释**以方便理解其代码实现，详见[Co-SLAM_Scene_Representation_Noted](./Co-SLAM_Scene_Representation_Noted/) 和 [Co-SLAM_Tracking_Noted](./Co-SLAM_Tracking_Noted/)。
3. 更多**视频解析**链接如下。


## 视觉SLAM见解
* **NeRF**: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV, 2020. [[Paper Insights](./1.Nerf开篇论文解读%20.pdf)]  [[Paper](https://arxiv.org/pdf/2003.08934.pdf)] [[Tensorflow Code](https://github.com/bmild/nerf)] [[Webpage](http://tancik.com/nerf)] [[Video](https://www.youtube.com/watch?v=JuH79E8rdKc)] 
* **NICE-SLAM**: Neural Implicit Scalable Encoding for SLAM, CVPR, 2021. [[Code Notes](5.NICE-SLAM源码阅读笔记.pdf)] [[Tracking Insights](./6.NICE-SLAM跟踪代码解析和扩展内容.pdf)] [[Mapping Insights](./7.NICE-SLAM_Mapping.pdf)]  [[Paper](https://arxiv.org/abs/2112.12130)] [[Code](https://github.com/cvg/nice-slam)] [[Website](https://pengsongyou.github.io/nice-slam?utm_source=catalyzex.com)]
* **iMap**: Implicit Mapping and Positioning in Real-Time, ICCV, 2021. [[Paper Insights](./2.iMap解读.pdf)] [[Paper](https://arxiv.org/abs/2103.12352)] [[Website](https://edgarsucar.github.io/iMAP/)] [[Video](https://www.youtube.com/watch?v=c-zkKGArl5Y)]
*  **NICER-SLAM**: Neural Implicit Scene Encoding for RGB SLAM, arXiv, 2023. [[Paper Insights](./4.NICER-SLAM论文解读.pdf)]   [[Paper](https://arxiv.org/pdf/2302.03594.pdf)] [[Video](https://www.youtube.com/watch?v=tUXzqEZWg2w)]
*  **Co-SLAM**: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM, CVPR, 2023. [[Mapping Insights](./10.Co-SLAM_Mapping.pdf)]  [[Paper](https://arxiv.org/pdf/2304.14377.pdf)] [[Website](https://hengyiwang.github.io/projects/CoSLAM)]
* **NeRF-SLAM**: Real-Time Dense Monocular SLAM with Neural Radiance Fields, arXiv, 2022. [[Paper Insights](./13.NeRF-SLAM论文框架梳理_Real-Time%20Dense%20Monocular%20SLAM%20with%20Neural%20Radiance%20Fields.pdf)]  [[Problem Record](./14-15.Nerf-Slam实践问题记录.docx)]     [[Paper](https://arxiv.org/pdf/2210.13641.pdf)] [[Pytorch Code](https://github.com/ToniRV/NeRF-SLAM)] [[Video](https://www.youtube.com/watch?v=-6ufRJugcEU)]
*  **vMAP**: Vectorised Object Mapping for Neural Field SLAM, CVPR,  2023. [[Paper Insights](./16.vMAP%20Vectorised%20Object%20Mapping%20for%20Neural%20Field%20SLAM.html)]  [[Paper](https://arxiv.org/pdf/2302.01838.pdf)] [[Website](https://kxhit.github.io/vMAP)] [[Pytorch Code](https://github.com/kxhit/vMAP)] [[Video](https://kxhit.github.io/media/vMAP/vmap_raw.mp4)]
*  **RO-MAP**: Real-Time Multi-Object Mapping with Neural Radiance Fields, RAL, 2023. [[Paper Insights](./18.RO-MAP%20Real-Time%20Multi-Object%20Mapping%20with%20Neural.html)]   [[Paper](https://ieeexplore.ieee.org/document/10209177)] [[Code](https://github.com/XiaoHan-Git/RO-MAP)] [[Video](https://www.youtube.com/watch?v=sFrLXPw40wU)]
*  Neural Implicit Dense Semantic SLAM, arXiv, 2023. [[Paper Insights](./17.Neural%20Implicit%20Dense%20Semantic%20SLAM.md)]   [[Paper](https://arxiv.org/pdf/2304.14560.pdf)]



## 激光SLAM见解
- Efficient Implicit Neural Reconstruction Using LiDAR, ICRA, 2023. [[Paper Insights](./19.Efficient%20Implicit%20Neural%20Reconstruction%20Using%20LiDAR论文框架梳理.pdf)]  [[Paper](https://arxiv.org/pdf/2302.14363.pdf)] [[Website](http://starydy.xyz/EINRUL/)] [[Pytorch Code](https://github.com/StarRealMan/EINRUL)] [[Video](https://www.youtube.com/watch?v=wUp2I-X-IdI)]
- **NeRF-LOAM**: Neural Implicit Representation for Large-Scale Incremental LiDAR Odometry and Mapping, ICCV, 2023. [[Paper Insights](./12.基于Lidar的NeRF-LOAM论文解读.pdf)]    [[Paper](https://arxiv.org/pdf/2303.10709.pdf)] [[Code](https://github.com/JunyuanDeng/NeRF-LOAM)]



## 视频链接

* [[第01讲 田宇博-NeRF开篇论文解读 NeRF](https://t.zsxq.com/13Rdok77J)]
* [[第02讲 田宇博-第一个稠密的实时NeRF SLAM iMAP论文解读](https://t.zsxq.com/13NT9SrVd)]
* [[第03讲 刘权祥-NICE SLAM论文解读](https://t.zsxq.com/13p6PzgGA)]
* [[第04讲 NICER SLAM论文解读](https://t.zsxq.com/133LIwane)]
* [[第05讲（上）-刘权祥-NICE SLAM代码解读：整体代码框架及运行：跟踪](https://t.zsxq.com/13Mjh18d9)]
* [[第05讲（下）-刘权祥-NICE SLAM代码解读：整体代码框架及运行：跟踪](https://t.zsxq.com/13pr1Ka69)]
* [[第06讲（上）-汪寿安-NICE SLAM代码解读](https://t.zsxq.com/13EWNTdeZ)]
* [[第06讲（下）-汪寿安-NICE SLAM代码解读](https://t.zsxq.com/13cDZxv3a)]
* [[第07讲 NICE SLAM代码解读：建图](https://t.zsxq.com/13ZeZgo36)]
* [[第08讲 钟至德-Co-SLAM论文解读](https://t.zsxq.com/13yYcc3yp)]
* [[第09讲 徐扬-Co-SLAM 代码解读：tracking](https://t.zsxq.com/13MHRa6rH)]
* [[第10讲 Co-SLAM 代码解读：mapping](https://t.zsxq.com/13N9RJVaj)]
* [[第11讲 张一 Co-SLAM 代码解读：Scene representation](https://t.zsxq.com/13WlZnCY1)]
* [[第12讲（上）-汪寿安-基于LiDAR的NeRF-LOAM论文解读](https://t.zsxq.com/13BnX2HN4)]
* [[第12讲（中）-汪寿安-基于LiDAR的NeRF-LOAM论文解读](https://t.zsxq.com/13Lrj9ECe)]
* [[第12讲（下）-汪寿安-基于LiDAR的NeRF-LOAM论文解读](https://t.zsxq.com/13nImsTqq)]
* [[第13讲 张一 NeRF-SLAM 论文框架梳理](https://t.zsxq.com/13iv6vYgR)]
* [[第14-15讲 陈安东 NeRF-SLAM 运行配置经验](https://t.zsxq.com/13rkfR21n)]
* [[第16讲-夏宁宁-物体级vMAP 论文解读](https://t.zsxq.com/13Sl2SAPy)]
* [[第17讲-徐扬-语义Neural Implicit Dense Semantic SLAM 论文解读](https://t.zsxq.com/13MtOSTLz)]
* [[第18讲-夏宁宁-实时多物体RO-MAP 论文解读](https://t.zsxq.com/13fVUbp2w)]
* [[第19讲-基于LiDAR的Efficient Implicit Neural Reconstruction Using LiDAR 论文解读](https://t.zsxq.com/13Ofplkrr)]
* [[第20讲-LiDAR全局定位 IRMCL Implicit Representation-based Online Global Localization论文解读](https://t.zsxq.com/13kYwivPD)]

知识星球成员可免费观看视频

![zsxq](images/Life_Planet.JPG)



## 致谢
本项目来自于[计算机视觉life](https:cvlife.net)的“Nerf Based SLAM算法学习小组”，贡献成员包括（不区分先后顺序）：


田宇博、刘权祥、史慧、汪寿安、万静怡、钟志德、徐扬、张一、陈安东、夏宁宁



## 引用
```
@misc{electron2023nerfbasedslamincredibleinsights,
    title = {NeRF-Based-SLAM-Incredible-Insights},
    author = {electron6,shuttworth},
    journal = {GitHub repository},
    url = {https://github.com/electech6/NeRF-Based-SLAM-Incredible-Insights},
    year = {2023}
}
```


## Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=electech6/NeRF-Based-SLAM-Incredible-Insights&type=Date)](https://star-history.com/#electech6/NeRF-Based-SLAM-Incredible-Insights&Date)
