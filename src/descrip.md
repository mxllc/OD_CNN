# 结构
|      | Input size | Conv kernel size / pooling method | num | padding |stride | output size | parameters |
| ---- | :--: | ---- | ---- | ---- | ---- | ---- | ---- |
| Input | 116x116x18 | / | / | / | / | / | / |
| Conv1 | 116x116x18 | 11x11X18 | 96 | SAME | 2 | 55x55x96 | 209184 |
| Conv2 | 55x55x96 | 5x5x96 | 256 | SAME | 2 | 27x27x96 | 614656 |
| Pool1 | 27x27x96 | MaxPooling | 1 | SAME | 2 | 27x27x256 | / |
| Conv3 | 27x27x256 | 3x3x256 | 384 | SAME | 1 | 13x13x384 | 885120 |
| Pool2 | 13x13x384 | 2x2/MaxPooling | 1 | SAME | 2 | 13x13x384 | / |
| Conv4 | 13x13x384 | 3x3x384 | 384 | SAME | 1 | 13x13x256 | 1327488 |
| Conv5 | 13x13x256  | 3x3x256 | 256 | SAME | 1 | 4x4x256 | 590080 |
| Pool3 | 4x4x256 | 2x2/MaxPooling | 1 | SAME | 2 | 4x4x256 | / |
| FC1 | 4096x1 | / | 4096 | / | / | 4096x1 | 16777216 |
| FC2 | 4096x1 | / | 4096 | / | / | 4096x1 | 16777216 |
| FC3 | 4096x1 | / | 311 | / | / | 311x1 | 1273856 |


# loss函数
..|| L2||
#learning rate
...

# 改进
|      | Input size | Conv kernel size / pooling method | num | padding |stride | output size | parameters |
| ---- | :--: | ---- | ---- | ---- | ---- | ---- | ---- |
| Input | 227x227x3 | / | / | / | / | / | / |
| Conv1 | 227x227x3 | 11x11X3 | 96 | SAME | 4 | 55x55x96 | 34944 |
| Conv2 | 55x55x96 | 5x5x96 | 256 | SAME | 2 | 27x27x96 | 614656 |
| Pool1 | 27x27x96 | MaxPooling | 1 | SAME | 2 | 27x27x256 | / |
| Conv3 | 27x27x256 | 3x3x256 | 384 | SAME | 1 | 13x13x384 | 885120 |
| Pool2 | 13x13x384 | 2x2/MaxPooling | 1 | SAME | 2 | 13x13x384 | / |
| Conv4 | 13x13x384 | 3x3x384 | 384 | SAME | 1 | 13x13x256 | 1327488 |
| Conv5 | 13x13x256  | 3x3x256 | 256 | SAME | 1 | 4x4x256 | 590080 |
| Pool3 | 4x4x256 | 2x2/MaxPooling | 1 | SAME | 2 | 4x4x256 | / |
| FC1 | 4096x1 | / | 4096 | / | / | 4096x1 | 16777216 |
| FC2 | 4096x1 | / | 4096 | / | / | 4096x1 | 16777216 |
| FC3 | 4096x1 | / | 1000 | / | / | 1000x1 | 4096000 |


$\frac{1}{2}​$