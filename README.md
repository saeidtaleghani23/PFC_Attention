# PFC_Attention
In this repository, a new transformer model is implemented that utilizes fine- and coarse-grained self-attentions to classify different objects in Compact Polarimetric (CP) Synthetic Aperture RADAR (SAR) data.
Since the used data is not public data, I cannot to share it here. However, the mail core of the this repository is PFC model in the model folder.

# Abstract

Land cover classification from compact polarimetry (CP) imagery captured by the launched RADARSAT Constellation Mission (RCM) is important but challenging due to class signature ambiguity issues and speckle noise.

This paper presents a new land cover classification method to improve the learning of discriminative features based on a novel *pyramid fine- and coarse-grained self-attentions transformer* (PFC transformer).

The fine-grained dependency inside a non-overlapping window and coarse-grained dependencies between non-overlapping windows are explicitly modeled and concatenated using a learnable linear function. This process is repeated in a hierarchical manner. Finally, the output of each stage of the proposed method is spatially reduced and concatenated to take advantage of both low- and high-level features.

Two high-resolution (3m) RCM CP SAR scenes are used to evaluate the performance of the proposed method and compare it to other state-of-the-art deep learning methods. The results show that the proposed approach achieves an overall accuracy of 93.63\% which was 4.83\% higher than the best comparable method, demonstrating the effectiveness of the proposed approach for land cover classification from RCM CP SAR images.

# PFC Attention Paper

Please read PFC_Attention_paper.pdf for details of the proposed method.

