# Status
* Modified flow decoder to output both flow and flow variance (4 channel output)
* Tried some of the tricks described in: https://upcommons.upc.edu/bitstream/handle/2117/100566/122527.pdf?isAllowed=y&sequence=1
    - ELU() + 1 activation to ensure variance is always positive
    - Add small constant value to variance before applying log => avoid NaN issues
    - Use Laplacian output instead of Gaussian output (L1 norm instead of L2)

* With these modifications, the network does not converge properly. Min/max values of predictions have reasonable values, but the EPE does not decrease
    - The loss continues to decrease below 0

## Updates
* Prediction in log-space => get rid of log in loss function
    - Test if improves convergence
* Clamp variance to small value (ex. 1e-5) before log

Results: convergence has not improved; loss continues to decrease, while EPE increases
<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=7297bd99e76d4ae6bf572229328f3392&tasks=dd661d37636540db91bb3cc35ee2b011&tasks=3168985e2dca43a8a88a0b63ab932430&metrics=flow_loss&variants=flow_loss&company=0860b8efd3d540d0b7efea317016d705"
  name="2e3cea65-e7aa-4c3f-ab48-6b483d7d1f96"
  width="100%" height="400"
></iframe>

<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=dd661d37636540db91bb3cc35ee2b011&tasks=665c30ed645c4f188b90dcda813807ed&tasks=7297bd99e76d4ae6bf572229328f3392&metrics=epe&variants=epe&company=0860b8efd3d540d0b7efea317016d705"
  name="fb4ddb48-7c86-4f7e-ac3f-acae5b880b02"
  width="100%" height="400"
></iframe>

# Next steps
* Check EPE at different iterations (maybe just later iterations are affected?)
* Current approach is not entirely correct (?)
    - At iteration t, the decoder predicts residual flow df (f{t} = f{t-1} + df), but variance is predicted for f{t}, not for df
    - Maybe should predict variance of residual flow and add it to current estimation of variance
* Investigate uncertainty propagation in the GRU layer => do not only predict variance at each iteration, but propagate it between iterations
* Experiment with RAFT pre-trained on FlyingChairs
    - Maybe learning variance from the beginning is too difficult and must be done in the fine tuning stages
* Read this paper explaining epistemic and aleatoric uncertainty: https://papers.nips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf1


