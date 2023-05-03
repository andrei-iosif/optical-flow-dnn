## Baseline results
* RAFT trained on FlyingChairs and FlyingThings3D (model weights downloaded from official repo)
* Results: https://app.clear.ml/projects/0df5a4dd4c7f43dd9f99529a3b32c95f/experiments/6da4b6dcb27d412880a2acdd75289a74/output/execution
* Validation KITTI: EPE=4.998567, F1=17.342550
 

## Training on VIPER dataset
* Pre-trained model (Chairs+Things) fine tuned on VIPER (subset with every 10th frame pair)
* Results: https://app.clear.ml/projects/0df5a4dd4c7f43dd9f99529a3b32c95f/experiments/8e0a64b4114547b4aa4314f62545f5f4/output/execution
* Validation KITTI: EPE=2.147832, F1=7.725108


## Training on VirtualKITTI dataset
* Pre-trained model (Chairs+Things) fine tuned on VirtualKITTI
* Results: https://app.clear.ml/projects/0df5a4dd4c7f43dd9f99529a3b32c95f/experiments/aca0731624c24e35bfc3d09b5ff97647/output/execution
* Validation KITTI: EPE=2.565336, F1=7.941933


## Training on both VIPER and VirtualKITTI
* Pre-trained model (Chairs+Things) fine tuned on VIPER, then on VirtualKITTI
* Results: https://app.clear.ml/projects/0df5a4dd4c7f43dd9f99529a3b32c95f/experiments/58f65da0c71f43c9a61f16f1f1de8974/output/execution
* Validation KITTI: EPE=2.139484, F1=7.511944

## Visual results
Difference from KITTI: in VIPER dataset the flow on vehicles is not uniform. Not a huge issue, but may impact training with semantic loss
![18_viper_val.jpeg](https://app.clear.ml/files//reports/Training with different datasets.9e413e105f69412a8f7dcad640f972e8/18_viper_val.jpeg)


# TODO: 
* Repeat experiments with all 3 seeds => report metrics
* Investigate how to remove wrong flow estimation on sky
