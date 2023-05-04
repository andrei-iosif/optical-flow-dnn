## Initial training: Overfit on small subset of VIPER (100 samples)
* Base RAFT
* RAFT with semantic loss

Same crop size as KITTI training stage (288x960)

#### Initial results:
* Semantic loss does not seem to decrease very much, need to check again correctness.
* Also check semseg label resizing (what kind of interpolation is used)


#### Update
* Loss implementation is correct
* Modified label interpolation (from bilinear to nearest neighbor) -- needed for spatial augmentations (resize)
* Currently the semantic loss is computed only for valid GT flow values, taking into account the occlusions
    - Loss is computed for pixels with same semseg class in both frames


## Train baseline models (original loss)
* 3 different random seeds
    -  Train first with 1 seed, then decide if continue with other 2 (takes about 24h)
    -  UPDATE: trained with all 3 seeds
* Train on VIPER (do not mix for now, investigate later how to train both on VIPER and VirtualKITTI)
    -  Mixed precision training (shown that it does not crash due to memory issues)
    -  Validate on subset of VIPER val (100-200 samples); this time it's guaranteed that the validation samples are selected randomly
    

## Train with semantic loss (RAFT semantic V1)
* Same as before, 3 random seeds
* Same training and validation sets
* Compare results
    - For all 3 seeds, the performance with semantic loss starts to decrease after around 55k iterations (both training and validation metrics get worse); it seems that training becomes unstable and the network diverges
<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=a52863550e8042058a75bcd1da2548df&tasks=3832f260cf0d42e988d078d28931cc86&metrics=1px&variants=1px&company=0860b8efd3d540d0b7efea317016d705"
  name="3480a023-b1e9-4f8b-9f12-cd33e7d933c0"
  width="100%" height="400"
></iframe>
    - If we look at KITTI validation results after around 50K iterations, we see some improvement:
<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=a52863550e8042058a75bcd1da2548df&tasks=3832f260cf0d42e988d078d28931cc86&metrics=kitti-epe&variants=kitti-epe&company=0860b8efd3d540d0b7efea317016d705"
  name="2d74f2c9-7e8e-4a72-afd1-97ada734d6ea"
  width="100%" height="400"
></iframe>

<iframe
  src="https://app.clear.ml/widgets/?type=scalar&tasks=a52863550e8042058a75bcd1da2548df&tasks=3832f260cf0d42e988d078d28931cc86&metrics=kitti-f1&variants=kitti-f1&company=0860b8efd3d540d0b7efea317016d705"
  name="862b831b-2845-4f21-b994-3bdb187411b9"
  width="100%" height="400"
></iframe>


## Next steps
* Inference visus with the 2 versions
* Try training on Virtual KITTI, same scenarios (baseline and RAFT semantic v1)
* Investigate cause of training divergence
    - Take a look at loss value at different stages of flow refinement (maybe need to change number of refinement steps?)
* Modify architecture to add semseg decoder
    - May need further downsampling before this (?) - encoder downsamples only to 1/8 input resolution
    - Need to predict for both input images
    - Could add temporal consistency loss for semseg predictions


## Improvement ideas (OLD)
* Currently we use only the semseg mask for the first frame to compute the loss
    - Maybe we can benefit somehow from using both (?)
        - Add something similar to photometric consistency, but for semseg GT (warp first semseg mask using flow and compute difference with second mask)
        - Need to take into account occlusions -> need to also estimate an occlusion mask and compute consistency loss only for non-occluded pixels
        - For occlusion mask we need to compute backward flow (or estimate occlusion mask as additional output channel, similarly to https://openaccess.thecvf.com/content/CVPR2022/html/Jeong_Imposing_Consistency_for_Optical_Flow_Estimation_CVPR_2022_paper.html)