# High Level Operations Status
|    | Operations                     |   Input Variations |   Converted |   Removed |   Fallback | Completed   |   Score |
|---:|:-------------------------------|-------------------:|------------:|----------:|-----------:|:------------|--------:|
|  0 | aten._softmax.default          |                  1 |           0 |         1 |          0 | ✅          |     1   |
|  1 | aten._to_copy.default          |                  1 |           1 |         0 |          0 | ✅          |     1   |
|  2 | aten._unsafe_view.default      |                  1 |           1 |         0 |          0 | ✅          |     1   |
|  3 | aten.add.Tensor                |                  6 |           5 |         1 |          0 | ✅          |     1   |
|  4 | aten.addmm.default             |                  6 |           0 |         6 |          0 | ✅          |     1   |
|  5 | aten.bmm.default               |                  2 |           2 |         0 |          0 | ✅          |     1   |
|  6 | aten.clone.default             |                  4 |           0 |         4 |          0 | ✅          |     1   |
|  7 | aten.div.Tensor                |                  1 |           0 |         1 |          0 | ✅          |     1   |
|  8 | aten.embedding.default         |                  3 |           3 |         0 |          0 | ✅          |     1   |
|  9 | aten.expand.default            |                  3 |           0 |         3 |          0 | ✅          |     1   |
| 10 | aten.mul.Tensor                |                  9 |           9 |         0 |          0 | ✅          |     1   |
| 11 | aten.native_layer_norm.default |                  2 |           1 |         0 |          0 | 🚧          |     0.5 |
| 12 | aten.permute.default           |                  1 |           1 |         0 |          0 | ✅          |     1   |
| 13 | aten.pow.Tensor_Scalar         |                  2 |           2 |         0 |          0 | ✅          |     1   |
| 14 | aten.rsub.Scalar               |                  1 |           1 |         0 |          0 | ✅          |     1   |
| 15 | aten.slice.Tensor              |                  2 |           1 |         1 |          0 | ✅          |     1   |
| 16 | aten.t.default                 |                  6 |           0 |         6 |          0 | ✅          |     1   |
| 17 | aten.tanh.default              |                  2 |           2 |         0 |          0 | ✅          |     1   |
| 18 | aten.transpose.int             |                  2 |           2 |         0 |          0 | ✅          |     1   |
| 19 | aten.unsqueeze.default         |                  2 |           2 |         0 |          0 | ✅          |     1   |
| 20 | aten.view.default              |                 13 |          13 |         0 |          0 | ✅          |     1   |
***
### aten._softmax.default
|    | ATen Input Variations                                                          | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 64, 9, 9]> self = ?,<br>int dim = -1,<br>bool half_to_float = False | Removed  | Done       | 0.999596 |      0 |
### aten._to_copy.default
|    | ATen Input Variations                                                  | Status   | Isolated   |   PCC |   Host |
|---:|:-----------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 1, 1, 9]> self = ?,<br>Optional[int] dtype = torch.bfloat16 | Done     | Fallback   |     1 |     -1 |
### aten._unsafe_view.default
|    | ATen Input Variations                                             | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 9, 64, 64]> self = ?,<br>List[int] size = [1, 9, 4096] | Done     | Done       |     1 |      0 |
### aten.add.Tensor
|    | ATen Input Variations                                              | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 64, 9, 9]> self = ?,<br>Tensor<[1, 1, 1, 9]> other = ?  | Removed  | Done       | 0.999997 |      0 |
|  1 | Tensor<[1, 9, 128]> self = ?,<br>Tensor other = 1.0                | Done     | Done       | 0.999994 |      0 |
|  2 | Tensor<[1, 9, 128]> self = ?,<br>Tensor<[1, 9, 128]> other = ?     | Done     | Done       | 0.999998 |      0 |
|  3 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor other = 1.0              | Done     | Done       | 0.999995 |      0 |
|  4 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor<[1, 9, 16384]> other = ? | Done     | Done       | 0.999998 |      0 |
|  5 | Tensor<[1, 9, 4096]> self = ?,<br>Tensor<[1, 9, 4096]> other = ?   | Done     | Done       | 0.999998 |      0 |
### aten.addmm.default
|    | ATen Input Variations                                                                      | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[128]> self = ?,<br>Tensor<[9, 4096]> mat1 = ?,<br>Tensor<[4096, 128]> mat2 = ?     | Removed  | Done       | 0.999922 |      0 |
|  1 | Tensor<[16384]> self = ?,<br>Tensor<[9, 4096]> mat1 = ?,<br>Tensor<[4096, 16384]> mat2 = ? | Removed  | Done       | 0.999933 |      0 |
|  2 | Tensor<[30000]> self = ?,<br>Tensor<[9, 128]> mat1 = ?,<br>Tensor<[128, 30000]> mat2 = ?   | Removed  | Done       | 0.999979 |      0 |
|  3 | Tensor<[4096]> self = ?,<br>Tensor<[9, 128]> mat1 = ?,<br>Tensor<[128, 4096]> mat2 = ?     | Removed  | Done       | 0.999979 |      0 |
|  4 | Tensor<[4096]> self = ?,<br>Tensor<[9, 16384]> mat1 = ?,<br>Tensor<[16384, 4096]> mat2 = ? | Removed  | Done       | 0.999802 |      0 |
|  5 | Tensor<[4096]> self = ?,<br>Tensor<[9, 4096]> mat1 = ?,<br>Tensor<[4096, 4096]> mat2 = ?   | Removed  | Done       | 0.999933 |      0 |
### aten.bmm.default
|    | ATen Input Variations                                         | Status   | Isolated   |      PCC |   Host |
|---:|:--------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[64, 9, 64]> self = ?,<br>Tensor<[64, 64, 9]> mat2 = ? | Done     | Done       | 0.999986 |      0 |
|  1 | Tensor<[64, 9, 9]> self = ?,<br>Tensor<[64, 9, 64]> mat2 = ?  | Done     | Done       | 0.999992 |      0 |
### aten.clone.default
|    | ATen Input Variations                                                                     | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 64, 9, 9]> self = ?                                                            | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[1, 9, 128]> self = ?                                                              | Removed  | Done       |     1 |      0 |
|  2 | Tensor<[1, 9, 4096]> self = ?                                                             | Removed  | Done       |     1 |      0 |
|  3 | Tensor<[1, 9, 64, 64]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format | Removed  | Done       |     1 |      0 |
### aten.div.Tensor
|    | ATen Input Variations                                 | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 64, 9, 9]> self = ?,<br>Tensor other = 8.0 | Removed  | Done       |     1 |      0 |
### aten.embedding.default
|    | ATen Input Variations                                                                  | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[2, 128]> weight = ?,<br>Tensor<[1, 9]> indices = ?                             | Done     | Done       |     1 |      0 |
|  1 | Tensor<[30000, 128]> weight = ?,<br>Tensor<[1, 9]> indices = ?,<br>int padding_idx = 0 | Done     | Done       |     1 |      0 |
|  2 | Tensor<[512, 128]> weight = ?,<br>Tensor<[1, 9]> indices = ?                           | Done     | Done       |     1 |      0 |
### aten.expand.default
|    | ATen Input Variations                                               | Status   | Isolated   |   PCC |   Host |
|---:|:--------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 64, 64, 9]> self = ?,<br>List[int] size = [1, 64, 64, 9] | Removed  | Done       |     1 |     -1 |
|  1 | Tensor<[1, 64, 9, 64]> self = ?,<br>List[int] size = [1, 64, 9, 64] | Removed  | Done       |     1 |     -1 |
|  2 | Tensor<[1, 64, 9, 9]> self = ?,<br>List[int] size = [1, 64, 9, 9]   | Removed  | Done       |     1 |     -1 |
### aten.mul.Tensor
|    | ATen Input Variations                                                    | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 1, 1, 9]> self = ?,<br>Tensor other = -3.3895313892515355e+38 | Done     | Done       | 0.999999 |      0 |
|  1 | Tensor<[1, 9, 128]> self = ?,<br>Tensor other = 0.044715                 | Done     | Done       | 0.999996 |      0 |
|  2 | Tensor<[1, 9, 128]> self = ?,<br>Tensor other = 0.5                      | Done     | Done       | 1        |      0 |
|  3 | Tensor<[1, 9, 128]> self = ?,<br>Tensor other = 0.7978845608028654       | Done     | Done       | 0.999997 |      0 |
|  4 | Tensor<[1, 9, 128]> self = ?,<br>Tensor<[1, 9, 128]> other = ?           | Done     | Done       | 0.999997 |      0 |
|  5 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor other = 0.044715               | Done     | Done       | 0.999995 |      0 |
|  6 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor other = 0.5                    | Done     | Done       | 1        |      0 |
|  7 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor other = 0.7978845608028654     | Done     | Done       | 0.999997 |      0 |
|  8 | Tensor<[1, 9, 16384]> self = ?,<br>Tensor<[1, 9, 16384]> other = ?       | Done     | Done       | 0.999996 |      0 |
### aten.native_layer_norm.default
|    | ATen Input Variations                                                                                                                                                      | Status   | Isolated   |      PCC |   Host |
|---:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 9, 128]> input = ?,<br>List[int] normalized_shape = [128],<br>Optional[Tensor]<[128]> weight = ?,<br>Optional[Tensor]<[128]> bias = ?,<br>float eps = 1e-12     | Done     | Done       | 0.999422 |      3 |
|  1 | Tensor<[1, 9, 4096]> input = ?,<br>List[int] normalized_shape = [4096],<br>Optional[Tensor]<[4096]> weight = ?,<br>Optional[Tensor]<[4096]> bias = ?,<br>float eps = 1e-12 | None     | Fallback   | 1        |      0 |
### aten.permute.default
|    | ATen Input Variations                                             | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 9, 64, 64]> self = ?,<br>List[int] dims = [0, 2, 1, 3] | Done     | Done       |     1 |      0 |
### aten.pow.Tensor_Scalar
|    | ATen Input Variations                                    | Status   | Isolated   |      PCC |   Host |
|---:|:---------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 9, 128]> self = ?,<br>number exponent = 3.0   | Done     | Done       | 0.999995 |      0 |
|  1 | Tensor<[1, 9, 16384]> self = ?,<br>number exponent = 3.0 | Done     | Done       | 0.999996 |      0 |
### aten.rsub.Scalar
|    | ATen Input Variations                                | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 1, 1, 9]> self = ?,<br>number other = 1.0 | Done     | Done       | 0.999994 |      0 |
### aten.slice.Tensor
|    | ATen Input Variations                                                                                             | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 512]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807 | Removed  | Done       |     1 |     -1 |
|  1 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9                   | Done     | Done       |     1 |      0 |
### aten.t.default
|    | ATen Input Variations          | Status   | Isolated   |   PCC |   Host |
|---:|:-------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[128, 4096]> self = ?   | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[16384, 4096]> self = ? | Removed  | Done       |     1 |      0 |
|  2 | Tensor<[30000, 128]> self = ?  | Removed  | Done       |     1 |      0 |
|  3 | Tensor<[4096, 128]> self = ?   | Removed  | Done       |     1 |      0 |
|  4 | Tensor<[4096, 16384]> self = ? | Removed  | Done       |     1 |      0 |
|  5 | Tensor<[4096, 4096]> self = ?  | Removed  | Done       |     1 |      0 |
### aten.tanh.default
|    | ATen Input Variations          | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 9, 128]> self = ?   | Done     | Done       | 0.999945 |      0 |
|  1 | Tensor<[1, 9, 16384]> self = ? | Done     | Done       | 0.999942 |      0 |
### aten.transpose.int
|    | ATen Input Variations                                               | Status   | Isolated   |   PCC |   Host |
|---:|:--------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 64, 9, 64]> self = ?,<br>int dim0 = -1,<br>int dim1 = -2 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 64, 9, 64]> self = ?,<br>int dim0 = 2,<br>int dim1 = 1   | Done     | Done       |     1 |      0 |
### aten.unsqueeze.default
|    | ATen Input Variations                      | Status   | Isolated   |   PCC |   Host |
|---:|:-------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 1, 9]> self = ?,<br>int dim = 2 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 9]> self = ?,<br>int dim = 1    | Done     | Done       |     1 |      0 |
### aten.view.default
|    | ATen Input Variations                                             | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 64, 64, 9]> self = ?,<br>List[int] size = [64, 64, 9]  | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 64, 9, 64]> self = ?,<br>List[int] size = [64, 9, 64]  | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 64, 9, 9]> self = ?,<br>List[int] size = [64, 9, 9]    | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 9, 128]> self = ?,<br>List[int] size = [9, 128]        | Done     | Done       |     1 |      0 |
|  4 | Tensor<[1, 9, 16384]> self = ?,<br>List[int] size = [9, 16384]    | Done     | Done       |     1 |      0 |
|  5 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [1, 9, 64, 64] | Done     | Done       |     1 |      0 |
|  6 | Tensor<[1, 9, 4096]> self = ?,<br>List[int] size = [9, 4096]      | Done     | Done       |     1 |      0 |
|  7 | Tensor<[64, 9, 64]> self = ?,<br>List[int] size = [1, 64, 9, 64]  | Done     | Done       |     1 |      0 |
|  8 | Tensor<[64, 9, 9]> self = ?,<br>List[int] size = [1, 64, 9, 9]    | Done     | Done       |     1 |      0 |
|  9 | Tensor<[9, 128]> self = ?,<br>List[int] size = [1, 9, 128]        | Done     | Done       |     1 |      0 |
| 10 | Tensor<[9, 16384]> self = ?,<br>List[int] size = [1, 9, 16384]    | Done     | Done       |     1 |      0 |
| 11 | Tensor<[9, 30000]> self = ?,<br>List[int] size = [1, 9, 30000]    | Done     | Done       |     1 |      0 |
| 12 | Tensor<[9, 4096]> self = ?,<br>List[int] size = [1, 9, 4096]      | Done     | Done       |     1 |      0 |

