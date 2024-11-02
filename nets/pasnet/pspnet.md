# pooling_layers

```txt
Sequential(
  (0): Sequential(
    (0): AdaptiveAvgPool3d(output_size=64)
    (1): Conv3d(4, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
  )
  (1): Sequential(
    (0): AdaptiveAvgPool3d(output_size=32)
    (1): Conv3d(4, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
  )
  (2): Sequential(
    (0): AdaptiveAvgPool3d(output_size=16)
    (1): Conv3d(4, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (2): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
  )
  (3): Sequential(
    (0): AdaptiveAvgPool3d(output_size=8)
    (1): Conv3d(4, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (2): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
  )
)
```
