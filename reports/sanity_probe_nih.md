# Linear-probe sanity check (AUROC)

```
   encoder split      C  atelectasis  cardiomegaly  consolidation  effusion  pneumothorax  no_finding  macro5
  resnet50   val 0.0001        0.741         0.747          0.751     0.805         0.799       0.718   0.768
  resnet50  test 0.0001        0.671         0.730          0.671     0.729         0.768       0.682   0.714
    dinov2   val 0.0100        0.763         0.792          0.755     0.820         0.811       0.725   0.788
    dinov2  test 0.0100        0.702         0.788          0.690     0.764         0.809       0.692   0.750
biomedclip   val 1.0000        0.765         0.848          0.784     0.842         0.804       0.735   0.809
biomedclip  test 1.0000        0.713         0.843          0.710     0.781         0.806       0.704   0.771
  rad_dino   val 0.0010        0.822         0.896          0.801     0.876         0.893       0.771   0.858
  rad_dino  test 0.0010        0.764         0.882          0.739     0.811         0.879       0.725   0.815
```
