from apex import amp
# Added after model and optimizer construction
model, optimizer = amp.initialize(model, optimizer, flags...)
...
# loss.backward() changed to:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

opt_levels：

Recognized opt_levels are "O0", "O1", "O2", and "O3".

O0 and O3 are not true mixed precision, but they are useful for establishing accuracy and speed baselines, respectively.

O1 and O2 are different implementations of mixed precision. Try both, and see what gives the best speedup and accuracy for your model.
O0: FP32 training

Your incoming model should be FP32 already, so this is likely a no-op. O0 can be useful to establish an accuracy baseline.
Default properties set by O0:
cast_model_type=torch.float32
patch_torch_functions=False
keep_batchnorm_fp32=None (effectively, “not applicable,” everything is FP32)
master_weights=False
loss_scale=1.0


O1: Mixed Precision (recommended for typical use)

Patch all Torch functions and Tensor methods to cast their inputs according to a whitelist-blacklist model. Whitelist ops (for example, Tensor Core-friendly ops like GEMMs and convolutions) are performed in FP16. Blacklist ops that benefit from FP32 precision (for example, softmax) are performed in FP32. O1 also uses dynamic loss scaling, unless overridden.
Default properties set by O1:
cast_model_type=None (not applicable)
patch_torch_functions=True
keep_batchnorm_fp32=None (again, not applicable, all model weights remain FP32)
master_weights=None (not applicable, model weights remain FP32)
loss_scale="dynamic"


O2: “Almost FP16” Mixed Precision

O2 casts the model weights to FP16, patches the model’s forward method to cast input data to FP16, keeps batchnorms in FP32, maintains FP32 master weights, updates the optimizer’s param_groups so that the optimizer.step() acts directly on the FP32 weights (followed by FP32 master weight->FP16 model weight copies if necessary), and implements dynamic loss scaling (unless overridden). Unlike O1, O2 does not patch Torch functions or Tensor methods.
Default properties set by O2:
cast_model_type=torch.float16
patch_torch_functions=False
keep_batchnorm_fp32=True
master_weights=True
loss_scale="dynamic"


O3: FP16 training

O3 may not achieve the stability of the true mixed precision options O1 and O2. However, it can be useful to establish a speed baseline for your model, against which the performance of O1 and O2 can be compared. If your model uses batch normalization, to establish “speed of light” you can try O3 with the additional property override keep_batchnorm_fp32=True (which enables cudnn batchnorm, as stated earlier).
Default properties set by O3:
cast_model_type=torch.float16
patch_torch_functions=False
keep_batchnorm_fp32=False
master_weights=False
loss_scale=1.0


