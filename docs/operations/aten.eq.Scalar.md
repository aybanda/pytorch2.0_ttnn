### aten.eq.Scalar
|    | ATen Input Variations                                    | Status   | Isolated   | PCC   | Host   |
|---:|:---------------------------------------------------------|:---------|:-----------|:------|:-------|
|  0 | Tensor<[1, 1, 1, 32]> self = ?,<br>number other = 0.0    | Unknown  | Unknown    | N/A   | N/A    |
|  1 | Tensor<[1, 1, 2048, 32]> self = ?,<br>number other = 0.0 | Unknown  | Unknown    | N/A   | N/A    |
|  2 | Tensor<[1, 16]> self = ?,<br>number other = 0            | Done     | Done       | 1.0   | 0      |
|  3 | Tensor<[1, 1]> self = ?,<br>number other = 0             | Done     | Done       | 1.0   | 0      |
|  4 | Tensor<[1, 45]> self = ?,<br>number other = 0            | Unknown  | Unknown    | N/A   | N/A    |
|  5 | Tensor<[1, 5]> self = ?,<br>number other = 0             | Unknown  | Unknown    | N/A   | N/A    |
|  6 | Tensor<[1, 7]> self = ?,<br>number other = 1             | Unknown  | Unknown    | N/A   | N/A    |
|  7 | Tensor<[1, 7]> self = ?,<br>number other = 50256         | Done     | Done       | 1.0   | 0      |
|  8 | Tensor<[1, s0]> self = ?,<br>number other = 0            | Unknown  | Unknown    | N/A   | N/A    |
|  9 | Tensor<[16, 49, 49]> self = ?,<br>number other = 0       | Done     | Done       | 1.0   | 0      |
| 10 | Tensor<[16, 64, 64]> self = ?,<br>number other = 0       | Done     | Done       | 1.0   | 0      |
| 11 | Tensor<[1]> self = ?,<br>number other = 1                | Unknown  | Done       | 1.0   | 0      |
| 12 | Tensor<[1]> self = ?,<br>number other = 50256            | Unknown  | Done       | 1.0   | 0      |
| 13 | Tensor<[2, 1]> self = ?,<br>number other = 0             | Done     | Done       | 1.0   | 0      |
| 14 | Tensor<[4, 49, 49]> self = ?,<br>number other = 0        | Done     | Done       | 1.0   | 0      |
| 15 | Tensor<[4, 64, 64]> self = ?,<br>number other = 0        | Done     | Done       | 1.0   | 0      |
| 16 | Tensor<[64, 49, 49]> self = ?,<br>number other = 0       | Done     | Done       | 1.0   | 0      |
| 17 | Tensor<[64, 64, 64]> self = ?,<br>number other = 0       | Done     | Done       | 1.0   | 0      |

