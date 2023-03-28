Datasets (see also https://github.com/andrei-iosif/optical-flow-dnn/blob/main/docs/flow_datasets.md):
* Pre-train on FlyingChairs and FlyingThings3D
* Train on:
    - Virtual KITTI
    - VIPER
    - HD1K (no semantic labels)
* Validate on:
    - VIPER validation set
* Evaluate on:
    - KITTI
	- VIPER test set
    - Sintel (?) -- maybe not relevant


Improvement ideas:
1. Add semseg correction to RAFT training
    * Additional loss term that enforces consistency between flow map and semantic map
        - Need to have similar boundaries
    * 2 variations:
        - Semantic RAFT V1: semantic consistency loss term
        - Semantic RAFT V2: add semseg prediction head + semseg loss + semantic consistency loss
    * Compare baseline RAFT with the 2 versions of semantic RAFT
    * Something similar to flow smoothness term (self-supervised scenario): https://arxiv.org/abs/2006.04902
    * Same idea was applied to mono depth: https://arxiv.org/abs/1810.04093 (and others)

2. Uncertainty estimation for optical flow
	* Research approaches
	    - Adapt approach used for FlowNet: https://arxiv.org/abs/1802.07095
	    - Another one: https://arxiv.org/pdf/1805.11327.pdf
	    - Classic approach: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wannenwetsch_ProbFlow_Joint_Optical_ICCV_2017_paper.pdf
    * Have uncertainty GT from HD1K dataset

3. Self-supervised RAFT
	* Train base model
	    - Maybe use some of the tricks described here: https://arxiv.org/abs/2105.07014
	* Add semantic constraints (?)
	
4. Synthetic -> real dataset generalization 
	* Train on:
		- Chairs, Things, Virtual KITTI, VIPER
		- Virtual KITTI and VIPER only
	* Evaluate on KITTI, VIPER
	* Is pretraining on Chairs and Things datasets really necessary for driving scenarios?