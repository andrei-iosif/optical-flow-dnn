# RAFT debugging
## Check if EPE is computed correctly vs loss (how the mask is applied)
* Loss is not computed correctly in original implementation (mask is applied, but average is computed over all pixels, not just the valid ones)
* https://github.com/andrei-iosif/optical-flow-dnn/blob/add-semseg-correction-for-raft/raft/core/test/test_losses.ipynb
* Check if this impacts training
    * Fine tune raft-things on small subset of VIPER with both versions of loss (original and fixed)
    * Does not seem to impact performance (loss is obviously larger than before)

<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=b90554aa7441489b9c13e732392036f2&tasks=0afe9b5ca95340eaa792276b9073386b&metrics=flow_loss&variants=flow_loss&company=0860b8efd3d540d0b7efea317016d705"
  name="f61cfcd8-fb43-4faf-8ae5-8e1d3fd046fe"
  width="100%" height="400"
></iframe>

<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=b90554aa7441489b9c13e732392036f2&tasks=0afe9b5ca95340eaa792276b9073386b&metrics=epe&variants=epe&company=0860b8efd3d540d0b7efea317016d705"
  name="d8b40c89-c0f1-4170-b3cc-2028afdc9182"
  width="100%" height="400"
></iframe>

<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=b90554aa7441489b9c13e732392036f2&tasks=0afe9b5ca95340eaa792276b9073386b&metrics=kitti-epe&variants=kitti-epe&company=0860b8efd3d540d0b7efea317016d705"
  name="23eb4136-2979-493a-8d25-dd6d5315d2cd"
  width="100%" height="400"
></iframe>



## Investigate semseg divergence issue 
* Check if semseg image number is consistent with flow image - looks OK (see https://github.com/andrei-iosif/optical-flow-dnn/blob/add-semseg-correction-for-raft/raft/core/test/viper_dataset_log.txt)
* Train on smaller subset
* Print image numbers when network start to diverge - TODO
    
### Results
* After training on smaller subset of VIPER, the metrics for flows from intermediate iterations look as expected (similar evolution over time, and error decrease with each iteration)
* Issue: after changing how mask is applied during loss computation, the flow loss has increased more than expected (just for RAFT-Semantic)
* Possible solutions: decreasing the learning rate, as the loss is larger now; tuning the parameter controlling the contribution of semantic loss to the total loss
* UPDATE: logging the intermediate metrics does not influence training (there are no issues related to gradients propagating from these)


## Pre-training
* Variants:
    - VIPER - done; evaluation (visual, KITTI train - do not have GT for test set)
    - Virtual KITTI - done
    - VIPER, then Virtual KITTI - done
    - Virtual KITTI, then VIPER - ?
    - VIPER + Virtual KITTI - ?
* Visual results on Drive
* Metrics in the other report
* Main issue observed is wrong values on sky (need to check old RAFT inference results on Bosch data; similar behavior); this happens only after training with Virtual KITTI

## TODO
* Uncertainty
    * Residual uncertainty
    * Uncertainty only at last iteration (separate decoder)