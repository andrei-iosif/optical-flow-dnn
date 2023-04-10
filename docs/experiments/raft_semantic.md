## Initial training: Overfit on small subset of VIPER (100 samples)
* Base RAFT
* RAFT with semantic loss

Same crop size as KITTI training stage (288x960)

Results:
Semantic loss does not seem to decrease very much, need to check again correctness.
Also check semseg label resizing (what kind of interpolation is used)



## Train baseline models (original loss)
* 3 different random seeds
    -  Train first with 1 seed, then decide if continue with other 2 (takes about 24h)
* Train on VIPER (do not mix for now, investigate later how to train both on VIPER and VirtualKITTI)
    -  Mixed precision training (shown that it does not crash due to memory issues)
    -  Validate on subset of VIPER val (100-200 samples); this time it's guaranteed that the validation samples are selected randomly
    -

## Train with semantic loss (RAFT semantic V1)
* Same as before, 3 random seeds
* Same training and validation sets
* Compare results