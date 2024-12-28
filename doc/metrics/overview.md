# DD-Ranking Metrics

DD-Ranking provides a set of metrics to evaluate the real informativeness of datasets distilled by different methods. We categorize dataset distillation methods by the type of labels they use: hard label and soft label. For each label type, we provide a evaluation class that computes hard label recovery (HLR), improvement over random (IOR), and the DD-Ranking score. Additionally, we provide a general evaluation class, integrating most of exisiting evaluation methods, that computes the traditional test accuracy.

* [HardLabelEvaluator](hard-label.md) computes HLR, IOR, and DD-Ranking score for methods using hard labels.
* [SoftLabelEvaluator](soft-label.md) computes HLR, IOR, and DD-Ranking score for methods using soft labels.
* [GeneralEvaluator](general.md) computes the traditional test accuracy for existing methods.
