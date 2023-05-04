# Goal
Adapt RAFT architecture so that it can estimate both optical flow and its uncertainty. At this point we do not care about improving performance, we want an optical flow confidence estimate.

Starting points:
* Adapt approach used for FlowNet: https://arxiv.org/abs/1802.07095
* Another one for FlowNet: https://arxiv.org/pdf/1805.11327.pdf
    - Implementation: https://github.com/ezjong/lightprobnets
* Classic approach: https://openaccess.thecvf.com/content_ICCV_2017/papers/Wannenwetsch_ProbFlow_Joint_Optical_ICCV_2017_paper.pdf
 

Evaluation:
* Visual evaluation on Sintel, KITTI, VIPER, etc.
* Use GT from HD1K dataset
 
Steps:
* Add log-likelihood loss -> predict parameters of probability distribution instead of point estimates for flow
    - Main idea of modified loss: there are two ways for the network to decrease the loss; either improve the mean or predict a high variance.
    - V1: Predict parameters at each refinement step, and propagate only the flow, not the uncertainty, through the GRU layer
    - V2: Propagate both flow and uncertainty in the refinement steps
* Train on Chairs and Things

Maximum likelihood estimation:
* http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
* https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/19/01-Maximum-likelihood-estimation.html

Mixture density networks
* https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
* https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
* https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
* https://stats.stackexchange.com/questions/347431/do-neural-networks-learn-a-function-or-a-probability-density-function


