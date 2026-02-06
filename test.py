import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(
    data="coco8.yaml", 
    epochs=10, 
    imgsz=640,
    teacher_model= "E:/00.MOTHER_BOXES/code/modification/yolo11_2x.yaml", #teacher model path for distillation
    mask_type = "pyramid", # original or pyramid
    # distillation of classification head
    cls_dist = True, # distills cls with bce if we set dist_kl to False
    cls_dist_kl = False, # distills cls with kl divergence if true else bce distillation
    cls_fg_mask = True , # Set to True to apply spatial foreground mask, False to disable
    cls_dist_t= 1.,# cls temperature parameter
    cls_alpha= 1., # cls distillation loss multiplier
    # distillation of bounding box head
    dfl_dist = True, # distilling directly dfl head for bb
    dfl_fg_mask = True,
    dfl_t= 1.,
    dfl_alpha= 1.,
    # our proposed m2d2 multimodal distillation for dfl heads internally poses the mask so we should first set the mask
    M2D2 = True,
    m2d2_t= 1.,
    m2d2_alpha= 1.,
    # distilling bounding box head using regression first converts dfl to continouos values then distills it
    l2_dist = True,
    l2_fg_mask = False,  # Set to False to disable spatial masking
    l2_alpha= 1., # regression head distillation loss mutiplier
    # distilling neck features in different forms from raw features to attention based and masked by TAL
    feat_distill = True ,# to enable neck features return and avoid code breaking
    feat = True, # calculate whole feature maps and supports masking
    feat_att = True, # calculates attention maps and supports masking
    feat_mask = True, # True if we want to apply mask to feat and feat_att and other methods of distillation listed bellow
    loss_ty = "cosine", #"l2 or cosine when feat or feat_att is set to true"
    feat_oth = True, #"following methods could be applied if this one set to true"
    # choose only one of the followings
    use_cwd= True, # Channel-Wise Distillation (CWD)
    use_crd= False, # Correlation / Relational Distillation (CRD-style)
    use_mmd= False, # Feature Distribution Matching (MMD)
    use_spatial_att= False, # Spatial Softmax Attention Distillation
    use_channel_att= False, # Channel Attention Distillation (SE-style)
    # hyper parameters for all feature level distillation
    level_weights= [1., 1., 1.], # channel weights for all feature distillation and cls distillation
    feature_lambda= 1., # feature loss weight multiplier
)



print("second")
model = YOLO("yolo11n.pt")
results = model.train(
    data="coco8.yaml", 
    epochs=1, 
    imgsz=640,
    teacher_model= "E:/00.MOTHER_BOXES/code/modification/yolo11_2x.yaml", #teacher model path for distillation
    mask_type = "pyramid", # original or pyramid
    # distillation of classification head
    cls_dist = True, # distills cls with bce if we set dist_kl to False
    cls_dist_kl = False, # distills cls with kl divergence if true else bce distillation
    cls_fg_mask = True , # Set to True to apply spatial foreground mask, False to disable
    cls_dist_t= 1.,# cls temperature parameter
    cls_alpha= 1., # cls distillation loss multiplier
    # distillation of bounding box head
    dfl_dist = True, # distilling directly dfl head for bb
    dfl_fg_mask = True,
    dfl_t= 1.,
    dfl_alpha= 1.,
    # our proposed m2d2 multimodal distillation for dfl heads internally poses the mask so we should first set the mask
    M2D2 = True,
    m2d2_t= 1.,
    m2d2_alpha= 1.,
    # distilling bounding box head using regression first converts dfl to continouos values then distills it
    l2_dist = True,
    l2_fg_mask = False,  # Set to False to disable spatial masking
    l2_alpha= 1., # regression head distillation loss mutiplier
    # distilling neck features in different forms from raw features to attention based and masked by TAL
    feat_distill = True ,# to enable neck features return and avoid code breaking
    feat = True, # calculate whole feature maps and supports masking
    feat_att = True, # calculates attention maps and supports masking
    feat_mask = True, # True if we want to apply mask to feat and feat_att and other methods of distillation listed bellow
    loss_ty = "cosine", #"l2 or cosine when feat or feat_att is set to true"
    feat_oth = True, #"following methods could be applied if this one set to true"
    # choose only one of the followings
    use_cwd= True, # Channel-Wise Distillation (CWD)
    use_crd= False, # Correlation / Relational Distillation (CRD-style)
    use_mmd= False, # Feature Distribution Matching (MMD)
    use_spatial_att= False, # Spatial Softmax Attention Distillation
    use_channel_att= False, # Channel Attention Distillation (SE-style)
    # hyper parameters for all feature level distillation
    level_weights= [1., 1., 1.], # channel weights for all feature distillation and cls distillation
    feature_lambda= 1., # feature loss weight multiplier
)