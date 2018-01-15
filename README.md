# unity-fast-doodle

1. We start with the same setup such as Generator feeding into VGG19.
2. The style loss will be defined by gram (if we follow online doodle)
3. The input image is noise + semantic map
  a. map is divided into CxHxW with C represending the number of distinct region.
4. For training we'd need to generate random maps (using diamond square -> KMeans clustering)

What's next?
- Take a heat map -> n color cluster -> separate into MxHxW tensor