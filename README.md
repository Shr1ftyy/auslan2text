# auslan2text #
App for translating Auslan to text

## :bulb: TODO: ##

 - Look more into dynamic gesture recognition
   - LS-HAN (and Hierarchical Attention Networks in general)
   - CNNs with recurrence
   - FlowNets for optical flow? 
 - Implementations must be lightweight enough to run well on mobile devices
   - Determined by amount of memory and FLOPs required to run inference model 
   - Look into different serialization formats (possibly looking at protobufs
	 for now)
 - Create and/or look for a larger and much more diverse training set for network
  - Implement data augmentation?
 - Make prettier GUI :sweat_smile: 
 
## Arhitecture results ##

About 0.09% loss after 20 epochs (simple CNN with Batch Norm. and Max Pooling, static fingerspelling ASL).
 - Currently only works with static input, no recurrence or dynamics
![20Epochs](./imgs/simple_cnn_results_20_epochs.png)


