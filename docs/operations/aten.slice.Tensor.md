### aten.slice.Tensor
|     | ATen Input Variations                                                                                                                      | Status   | Isolated   | PCC   | Host   |
|----:|:-------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|:------|:-------|
|   0 | Tensor self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                                    | Removed  | Unknown    | N/A   | N/A    |
|   1 | Tensor self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                                    | Unknown  | Unknown    | N/A   | N/A    |
|   2 | Tensor self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                                    | Unknown  | Unknown    | N/A   | N/A    |
|   3 | Tensor<[1, 1, 1, 10]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
|   4 | Tensor<[1, 1, 1, 15]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Unknown    | N/A   | N/A    |
|   5 | Tensor<[1, 1, 1, 16]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|   6 | Tensor<[1, 1, 1, 17]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|   7 | Tensor<[1, 1, 1, 17]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|   8 | Tensor<[1, 1, 1, 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|   9 | Tensor<[1, 1, 1, 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|  10 | Tensor<[1, 1, 1, 201]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Unknown    | N/A   | N/A    |
|  11 | Tensor<[1, 1, 1, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 46                                    | Unknown  | Done       | 1.0   | 0      |
|  12 | Tensor<[1, 1, 1, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 6                                     | Unknown  | Done       | 1.0   | 0      |
|  13 | Tensor<[1, 1, 1, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
|  14 | Tensor<[1, 1, 1, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int]<s10 + 1> end = ?                            | Unknown  | Failed     | N/A   | N/A    |
|  15 | Tensor<[1, 1, 1, 25]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
|  16 | Tensor<[1, 1, 1, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|  17 | Tensor<[1, 1, 1, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|  18 | Tensor<[1, 1, 1, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
|  19 | Tensor<[1, 1, 1, 45]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|  20 | Tensor<[1, 1, 1, 46]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|  21 | Tensor<[1, 1, 1, 59]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|  22 | Tensor<[1, 1, 1, 5]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|  23 | Tensor<[1, 1, 1, 60]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|  24 | Tensor<[1, 1, 1, 6]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
|  25 | Tensor<[1, 1, 1, 7]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
|  26 | Tensor<[1, 1, 1, s0 + 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
|  27 | Tensor<[1, 1, 1, s0 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
|  28 | Tensor<[1, 1, 1, s10 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
|  29 | Tensor<[1, 1, 1024, 1024]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Removed  | Done       | 1.0   | -1     |
|  30 | Tensor<[1, 1, 1024, 1024]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Removed  | Done       | 1.0   | -1     |
|  31 | Tensor<[1, 1, 1024, 1024]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 7                                  | Removed  | Done       | 1.0   | 0      |
|  32 | Tensor<[1, 1, 12, 16, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
|  33 | Tensor<[1, 1, 12, 16, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
|  34 | Tensor<[1, 1, 12, 16, 2]> self = ?,<br>int dim = 4,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
|  35 | Tensor<[1, 1, 12, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  36 | Tensor<[1, 1, 16, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  37 | Tensor<[1, 1, 16, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  38 | Tensor<[1, 1, 16, 32]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  39 | Tensor<[1, 1, 16, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2   | Unknown  | Done       | 1.0   | 0      |
|  40 | Tensor<[1, 1, 16, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2   | Unknown  | Done       | 1.0   | 0      |
|  41 | Tensor<[1, 1, 16, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  42 | Tensor<[1, 1, 16, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  43 | Tensor<[1, 1, 16, 64]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  44 | Tensor<[1, 1, 16, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                     | Unknown  | Done       | 1.0   | 0      |
|  45 | Tensor<[1, 1, 16, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 32,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
|  46 | Tensor<[1, 1, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | -1     |
|  47 | Tensor<[1, 1, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | -1     |
|  48 | Tensor<[1, 1, 17]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | -1     |
|  49 | Tensor<[1, 1, 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Unknown  | Done       | 1.0   | -1     |
|  50 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Done       | 1.0   | -1     |
|  51 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Done       | 1.0   | -1     |
|  52 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 45                                 | Unknown  | Done       | 1.0   | 0      |
|  53 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 5                                  | Unknown  | Done       | 1.0   | 0      |
|  54 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Done       | 1.0   | -1     |
|  55 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int] start = 45,<br>Optional[int] end = 46                                | Unknown  | Done       | 1.0   | 0      |
|  56 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int] start = 5,<br>Optional[int] end = 6                                  | Unknown  | Done       | 1.0   | 0      |
|  57 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 2,<br>Optional[int]<s10> start = ?,<br>Optional[int]<s10 + 1> end = ?                    | Unknown  | Failed     | N/A   | N/A    |
|  58 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                 | Unknown  | Unknown    | N/A   | N/A    |
|  59 | Tensor<[1, 1, 2048, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Done       | 1.0   | -1     |
|  60 | Tensor<[1, 1, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Unknown  | Done       | 1.0   | -1     |
|  61 | Tensor<[1, 1, 32, 32]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
|  62 | Tensor<[1, 1, 32, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
|  63 | Tensor<[1, 1, 32]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Removed  | Done       | 1.0   | -1     |
|  64 | Tensor<[1, 1, 384, 512]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Unknown  | Done       | 1.0   | -1     |
|  65 | Tensor<[1, 1, 384, 512]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Unknown  | Done       | 1.0   | -1     |
|  66 | Tensor<[1, 1, 40]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Removed  | Done       | 1.0   | -1     |
|  67 | Tensor<[1, 1, 45, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 45                                   | Unknown  | Done       | 1.0   | 0      |
|  68 | Tensor<[1, 1, 45, 45]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  69 | Tensor<[1, 1, 45, 45]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  70 | Tensor<[1, 1, 5, 2048]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 5                                     | Unknown  | Done       | 1.0   | 0      |
|  71 | Tensor<[1, 1, 51865]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
|  72 | Tensor<[1, 1, 59, 59]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  73 | Tensor<[1, 1, 59, 59]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
|  74 | Tensor<[1, 1, 7, 1024]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 7                                     | Removed  | Done       | 1.0   | 0      |
|  75 | Tensor<[1, 1, 7, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                      | Unknown  | Unknown    | N/A   | N/A    |
|  76 | Tensor<[1, 1, 7, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 32,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Unknown    | N/A   | N/A    |
|  77 | Tensor<[1, 1, 7, 7]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
|  78 | Tensor<[1, 1, 7, 7]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
|  79 | Tensor<[1, 1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Unknown    | N/A   | N/A    |
|  80 | Tensor<[1, 100, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
|  81 | Tensor<[1, 102, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                   | Done     | Done       | 1.0   | 0      |
|  82 | Tensor<[1, 102, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 14,<br>Optional[int] end = 38                                  | Done     | Done       | 1.0   | 0      |
|  83 | Tensor<[1, 102, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 38,<br>Optional[int] end = 102                                 | Done     | Done       | 1.0   | 0      |
|  84 | Tensor<[1, 1024, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                               | Done     | Done       | 1.0   | 0      |
|  85 | Tensor<[1, 1024, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                             | Done     | Unknown    | N/A   | N/A    |
|  86 | Tensor<[1, 1024, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 768                             | Done     | Unknown    | N/A   | N/A    |
|  87 | Tensor<[1, 1024, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 768,<br>Optional[int] end = 1024                            | Done     | Unknown    | N/A   | N/A    |
|  88 | Tensor<[1, 1024, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 384                                 | Done     | Done       | 1.0   | 0      |
|  89 | Tensor<[1, 1024, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 384,<br>Optional[int] end = 640                               | Done     | Done       | 1.0   | 0      |
|  90 | Tensor<[1, 1024, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 640,<br>Optional[int] end = 1024                              | Done     | Done       | 1.0   | 0      |
|  91 | Tensor<[1, 1024, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 640,<br>Optional[int] end = 896                               | Done     | Done       | 1.0   | 0      |
|  92 | Tensor<[1, 1024, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 896,<br>Optional[int] end = 1024                              | Done     | Done       | 1.0   | 0      |
|  93 | Tensor<[1, 1024, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 512                                 | Done     | Done       | 1.0   | 0      |
|  94 | Tensor<[1, 1024, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 1024                              | Done     | Done       | 1.0   | 0      |
|  95 | Tensor<[1, 1072, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 160                                   | Done     | Done       | 1.0   | 0      |
|  96 | Tensor<[1, 1072, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 432                                 | Done     | Done       | 1.0   | 0      |
|  97 | Tensor<[1, 1072, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 432,<br>Optional[int] end = 1072                                | Done     | Done       | 1.0   | 0      |
|  98 | Tensor<[1, 1088, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 512                                 | Done     | Done       | 1.0   | 0      |
|  99 | Tensor<[1, 1088, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 704                               | Done     | Done       | 1.0   | 0      |
| 100 | Tensor<[1, 1088, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 704,<br>Optional[int] end = 896                               | Done     | Done       | 1.0   | 0      |
| 101 | Tensor<[1, 1088, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 896,<br>Optional[int] end = 1088                              | Done     | Done       | 1.0   | 0      |
| 102 | Tensor<[1, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Removed  | Done       | 1.0   | -1     |
| 103 | Tensor<[1, 112, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 104 | Tensor<[1, 112, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 56                                   | Done     | Done       | 1.0   | 0      |
| 105 | Tensor<[1, 112, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 56,<br>Optional[int] end = 112                                 | Done     | Done       | 1.0   | 0      |
| 106 | Tensor<[1, 112, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 107 | Tensor<[1, 112, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 108 | Tensor<[1, 118, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 109 | Tensor<[1, 118, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 118                                 | Done     | Done       | 1.0   | 0      |
| 110 | Tensor<[1, 12, 1, 10]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 111 | Tensor<[1, 12, 1, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 112 | Tensor<[1, 12, 1, s0 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 113 | Tensor<[1, 12, 2, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 114 | Tensor<[1, 12, 2, 2]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 115 | Tensor<[1, 12, 2, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | 0      |
| 116 | Tensor<[1, 12, 3, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 117 | Tensor<[1, 12, 3, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 118 | Tensor<[1, 12, 3, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
| 119 | Tensor<[1, 12, s0 + 1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 120 | Tensor<[1, 12, s0 + 1, s0 + 1]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 121 | Tensor<[1, 12, s0 + 1, s0 + 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807          | Unknown  | Unknown    | N/A   | N/A    |
| 122 | Tensor<[1, 12, s0 + 2, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 123 | Tensor<[1, 12, s0 + 2, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 124 | Tensor<[1, 12, s0 + 2, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807              | Unknown  | Unknown    | N/A   | N/A    |
| 125 | Tensor<[1, 120, 160]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
| 126 | Tensor<[1, 120, 160]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
| 127 | Tensor<[1, 120, 28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 128 | Tensor<[1, 120, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 60                                   | Done     | Done       | 1.0   | 0      |
| 129 | Tensor<[1, 120, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 60,<br>Optional[int] end = 120                                 | Done     | Done       | 1.0   | 0      |
| 130 | Tensor<[1, 120, 28, 28]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 131 | Tensor<[1, 120, 28, 28]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 132 | Tensor<[1, 122, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 133 | Tensor<[1, 122, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 44                                  | Done     | Done       | 1.0   | 0      |
| 134 | Tensor<[1, 122, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 44,<br>Optional[int] end = 122                                 | Done     | Done       | 1.0   | 0      |
| 135 | Tensor<[1, 124, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                   | Done     | Done       | 1.0   | 0      |
| 136 | Tensor<[1, 124, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 14,<br>Optional[int] end = 28                                  | Done     | Done       | 1.0   | 0      |
| 137 | Tensor<[1, 124, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 28,<br>Optional[int] end = 42                                  | Done     | Done       | 1.0   | 0      |
| 138 | Tensor<[1, 124, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 42,<br>Optional[int] end = 56                                  | Done     | Done       | 1.0   | 0      |
| 139 | Tensor<[1, 124, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 56,<br>Optional[int] end = 124                                 | Done     | Done       | 1.0   | 0      |
| 140 | Tensor<[1, 128, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                 | Done     | Done       | 1.0   | 0      |
| 141 | Tensor<[1, 128, 128, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 128                               | Done     | Done       | 1.0   | 0      |
| 142 | Tensor<[1, 128, 224, 224]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                 | Done     | Done       | 1.0   | 0      |
| 143 | Tensor<[1, 128, 224, 224]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 128                               | Done     | Done       | 1.0   | 0      |
| 144 | Tensor<[1, 128, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                   | Done     | Done       | 1.0   | 0      |
| 145 | Tensor<[1, 128, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 128                                 | Done     | Done       | 1.0   | 0      |
| 146 | Tensor<[1, 1280, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 512                                   | Done     | Done       | 1.0   | 0      |
| 147 | Tensor<[1, 1280, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1024,<br>Optional[int] end = 1280                               | Done     | Done       | 1.0   | 0      |
| 148 | Tensor<[1, 1280, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 1024                                | Done     | Done       | 1.0   | 0      |
| 149 | Tensor<[1, 1280]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Unknown  | Done       | 1.0   | -1     |
| 150 | Tensor<[1, 1280]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Unknown  | Done       | 1.0   | -1     |
| 151 | Tensor<[1, 12]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 152 | Tensor<[1, 1370, 1280]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 153 | Tensor<[1, 1370, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Unknown    | N/A   | N/A    |
| 154 | Tensor<[1, 1370, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807                    | Removed  | Unknown    | N/A   | N/A    |
| 155 | Tensor<[1, 14, 14, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 156 | Tensor<[1, 14, 14, 256]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 157 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 158 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 159 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 160 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 161 | Tensor<[1, 14, 14, 512]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 162 | Tensor<[1, 14, 14, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 163 | Tensor<[1, 14, 14, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 164 | Tensor<[1, 14, 14, 512]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 165 | Tensor<[1, 14, 28, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 166 | Tensor<[1, 14, 28, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 167 | Tensor<[1, 14, 28, 256]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 168 | Tensor<[1, 14, 28, 256]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 169 | Tensor<[1, 142, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                   | Done     | Done       | 1.0   | 0      |
| 170 | Tensor<[1, 142, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 14,<br>Optional[int] end = 38                                  | Done     | Done       | 1.0   | 0      |
| 171 | Tensor<[1, 142, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 38,<br>Optional[int] end = 78                                  | Done     | Done       | 1.0   | 0      |
| 172 | Tensor<[1, 142, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 78,<br>Optional[int] end = 142                                 | Done     | Done       | 1.0   | 0      |
| 173 | Tensor<[1, 144, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 174 | Tensor<[1, 144, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 144                                 | Done     | Done       | 1.0   | 0      |
| 175 | Tensor<[1, 144, 768]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 176 | Tensor<[1, 1440, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 768                                   | Done     | Done       | 1.0   | 0      |
| 177 | Tensor<[1, 1440, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1216,<br>Optional[int] end = 1440                               | Done     | Done       | 1.0   | 0      |
| 178 | Tensor<[1, 1440, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 768,<br>Optional[int] end = 992                                 | Done     | Done       | 1.0   | 0      |
| 179 | Tensor<[1, 1440, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 992,<br>Optional[int] end = 1216                                | Done     | Done       | 1.0   | 0      |
| 180 | Tensor<[1, 1445, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 181 | Tensor<[1, 1445, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = -100,<br>Optional[int] end = 9223372036854775807                 | Done     | Done       | 1.0   | 0      |
| 182 | Tensor<[1, 145, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 183 | Tensor<[1, 145, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | 0      |
| 184 | Tensor<[1, 152, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 185 | Tensor<[1, 152, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 54                                  | Done     | Done       | 1.0   | 0      |
| 186 | Tensor<[1, 152, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 54,<br>Optional[int] end = 152                                 | Done     | Done       | 1.0   | 0      |
| 187 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 192                                   | Done     | Done       | 1.0   | 0      |
| 188 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                   | Done     | Done       | 1.0   | 0      |
| 189 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1280,<br>Optional[int] end = 1536                               | Done     | Done       | 1.0   | 0      |
| 190 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 192,<br>Optional[int] end = 512                                 | Done     | Done       | 1.0   | 0      |
| 191 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 768                                 | Done     | Done       | 1.0   | 0      |
| 192 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 1536                                | Done     | Done       | 1.0   | 0      |
| 193 | Tensor<[1, 1536, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 768,<br>Optional[int] end = 1280                                | Done     | Done       | 1.0   | 0      |
| 194 | Tensor<[1, 156, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 195 | Tensor<[1, 156, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 156                                 | Done     | Done       | 1.0   | 0      |
| 196 | Tensor<[1, 15]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Unknown    | N/A   | N/A    |
| 197 | Tensor<[1, 16, 1, 10]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 198 | Tensor<[1, 16, 1, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 199 | Tensor<[1, 16, 1, s0 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 200 | Tensor<[1, 16, 112, 112]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 201 | Tensor<[1, 16, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 8                                   | Done     | Done       | 1.0   | 0      |
| 202 | Tensor<[1, 16, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 8,<br>Optional[int] end = 16                                  | Done     | Done       | 1.0   | 0      |
| 203 | Tensor<[1, 16, 112, 112]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 204 | Tensor<[1, 16, 112, 112]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 205 | Tensor<[1, 16, 16, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 206 | Tensor<[1, 16, 16, 256]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 207 | Tensor<[1, 16, 16, 384]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 208 | Tensor<[1, 16, 16, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 209 | Tensor<[1, 16, 16, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 210 | Tensor<[1, 16, 16, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 211 | Tensor<[1, 16, 16, 512]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 212 | Tensor<[1, 16, 16, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 213 | Tensor<[1, 16, 16, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 214 | Tensor<[1, 16, 16, 512]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 215 | Tensor<[1, 16, 2, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 216 | Tensor<[1, 16, 2, 2]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 217 | Tensor<[1, 16, 2, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | 0      |
| 218 | Tensor<[1, 16, 3, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 219 | Tensor<[1, 16, 3, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 220 | Tensor<[1, 16, 3, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
| 221 | Tensor<[1, 16, 32, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 222 | Tensor<[1, 16, 32, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 223 | Tensor<[1, 16, 32, 256]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 224 | Tensor<[1, 16, 32, 256]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 225 | Tensor<[1, 16, s0 + 1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 226 | Tensor<[1, 16, s0 + 1, s0 + 1]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 227 | Tensor<[1, 16, s0 + 1, s0 + 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807          | Unknown  | Unknown    | N/A   | N/A    |
| 228 | Tensor<[1, 16, s0 + 2, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 229 | Tensor<[1, 16, s0 + 2, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 230 | Tensor<[1, 16, s0 + 2, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807              | Unknown  | Unknown    | N/A   | N/A    |
| 231 | Tensor<[1, 160, 7, 7]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 232 | Tensor<[1, 160, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 80                                     | Done     | Done       | 1.0   | 0      |
| 233 | Tensor<[1, 160, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 80,<br>Optional[int] end = 160                                   | Done     | Done       | 1.0   | 0      |
| 234 | Tensor<[1, 160, 7, 7]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 235 | Tensor<[1, 160, 7, 7]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 236 | Tensor<[1, 160, 73, 73]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                   | Done     | Done       | 1.0   | 0      |
| 237 | Tensor<[1, 160, 73, 73]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 160                                 | Done     | Done       | 1.0   | 0      |
| 238 | Tensor<[1, 160]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | -1     |
| 239 | Tensor<[1, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 240 | Tensor<[1, 172, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 241 | Tensor<[1, 172, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 44                                  | Done     | Done       | 1.0   | 0      |
| 242 | Tensor<[1, 172, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 44,<br>Optional[int] end = 172                                 | Done     | Done       | 1.0   | 0      |
| 243 | Tensor<[1, 17]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 244 | Tensor<[1, 184, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 245 | Tensor<[1, 184, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 92                                   | Done     | Done       | 1.0   | 0      |
| 246 | Tensor<[1, 184, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 92,<br>Optional[int] end = 184                                 | Done     | Done       | 1.0   | 0      |
| 247 | Tensor<[1, 184, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 248 | Tensor<[1, 184, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 249 | Tensor<[1, 185, 28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 250 | Tensor<[1, 185, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -57,<br>Optional[int] end = 9223372036854775807                | Done     | Done       | 1.0   | 0      |
| 251 | Tensor<[1, 185, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 128                                  | Done     | Done       | 1.0   | 0      |
| 252 | Tensor<[1, 185, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 147                                | Done     | Done       | 1.0   | 0      |
| 253 | Tensor<[1, 185, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 147,<br>Optional[int] end = 185                                | Done     | Done       | 1.0   | 0      |
| 254 | Tensor<[1, 192, 71, 71]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 96                                   | Done     | Done       | 1.0   | 0      |
| 255 | Tensor<[1, 192, 71, 71]> self = ?,<br>int dim = 1,<br>Optional[int] start = 96,<br>Optional[int] end = 192                                 | Done     | Done       | 1.0   | 0      |
| 256 | Tensor<[1, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Done       | 1.0   | -1     |
| 257 | Tensor<[1, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Done       | 1.0   | -1     |
| 258 | Tensor<[1, 197, 1024]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 259 | Tensor<[1, 197, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
| 260 | Tensor<[1, 197, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 1                                       | Done     | Done       | 1.0   | 0      |
| 261 | Tensor<[1, 197, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 197                                     | Done     | Done       | 1.0   | 0      |
| 262 | Tensor<[1, 2, 120, 160]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 263 | Tensor<[1, 2, 30, 40]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 264 | Tensor<[1, 2, 60, 80]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 265 | Tensor<[1, 200, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 266 | Tensor<[1, 200, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 100                                  | Done     | Done       | 1.0   | 0      |
| 267 | Tensor<[1, 200, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 100,<br>Optional[int] end = 200                                | Done     | Done       | 1.0   | 0      |
| 268 | Tensor<[1, 200, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 269 | Tensor<[1, 200, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 270 | Tensor<[1, 201, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Unknown    | N/A   | N/A    |
| 271 | Tensor<[1, 201]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Unknown    | N/A   | N/A    |
| 272 | Tensor<[1, 2048]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Removed  | Done       | 1.0   | -1     |
| 273 | Tensor<[1, 218, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 274 | Tensor<[1, 218, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 44                                  | Done     | Done       | 1.0   | 0      |
| 275 | Tensor<[1, 218, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 44,<br>Optional[int] end = 90                                  | Done     | Done       | 1.0   | 0      |
| 276 | Tensor<[1, 218, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 90,<br>Optional[int] end = 218                                 | Done     | Done       | 1.0   | 0      |
| 277 | Tensor<[1, 23, 40, 128]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Unknown    | N/A   | N/A    |
| 278 | Tensor<[1, 23, 40, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Unknown    | N/A   | N/A    |
| 279 | Tensor<[1, 23, 40, 128]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Unknown    | N/A   | N/A    |
| 280 | Tensor<[1, 23, 40, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Unknown    | N/A   | N/A    |
| 281 | Tensor<[1, 23, 40, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Unknown    | N/A   | N/A    |
| 282 | Tensor<[1, 23, 40]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Unknown    | N/A   | N/A    |
| 283 | Tensor<[1, 23, 40]> self = ?,<br>int dim = 1,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                      | Done     | Done       | 1.0   | 0      |
| 284 | Tensor<[1, 23, 40]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 285 | Tensor<[1, 23, 40]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                      | Done     | Done       | 1.0   | 0      |
| 286 | Tensor<[1, 23, 40]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Unknown    | N/A   | N/A    |
| 287 | Tensor<[1, 236, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 288 | Tensor<[1, 236, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 236                                 | Done     | Done       | 1.0   | 0      |
| 289 | Tensor<[1, 24, 56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 290 | Tensor<[1, 24, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 12                                    | Done     | Done       | 1.0   | 0      |
| 291 | Tensor<[1, 24, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 12,<br>Optional[int] end = 24                                   | Done     | Done       | 1.0   | 0      |
| 292 | Tensor<[1, 24, 56, 56]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 293 | Tensor<[1, 24, 56, 56]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 294 | Tensor<[1, 240, 28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 295 | Tensor<[1, 240, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 120                                  | Done     | Done       | 1.0   | 0      |
| 296 | Tensor<[1, 240, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 120,<br>Optional[int] end = 240                                | Done     | Done       | 1.0   | 0      |
| 297 | Tensor<[1, 240, 28, 28]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 298 | Tensor<[1, 240, 28, 28]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 299 | Tensor<[1, 25, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
| 300 | Tensor<[1, 256, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 128                                | Done     | Done       | 1.0   | 0      |
| 301 | Tensor<[1, 256, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 256                              | Done     | Done       | 1.0   | 0      |
| 302 | Tensor<[1, 256, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 128                                  | Done     | Done       | 1.0   | 0      |
| 303 | Tensor<[1, 256, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 256                                | Done     | Done       | 1.0   | 0      |
| 304 | Tensor<[1, 256, 64, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 128                                  | Done     | Done       | 1.0   | 0      |
| 305 | Tensor<[1, 256, 64, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 256                                | Done     | Done       | 1.0   | 0      |
| 306 | Tensor<[1, 257, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Unknown    | N/A   | N/A    |
| 307 | Tensor<[1, 25]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Removed  | Done       | 1.0   | -1     |
| 308 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 309 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 112,<br>Optional[int] end = 128                                | Done     | Done       | 1.0   | 0      |
| 310 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 262                                | Done     | Done       | 1.0   | 0      |
| 311 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 32                                  | Done     | Done       | 1.0   | 0      |
| 312 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 32,<br>Optional[int] end = 48                                  | Done     | Done       | 1.0   | 0      |
| 313 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 48,<br>Optional[int] end = 64                                  | Done     | Done       | 1.0   | 0      |
| 314 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 80                                  | Done     | Done       | 1.0   | 0      |
| 315 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 80,<br>Optional[int] end = 96                                  | Done     | Done       | 1.0   | 0      |
| 316 | Tensor<[1, 262, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 96,<br>Optional[int] end = 112                                 | Done     | Done       | 1.0   | 0      |
| 317 | Tensor<[1, 276, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 318 | Tensor<[1, 276, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 276                                 | Done     | Done       | 1.0   | 0      |
| 319 | Tensor<[1, 28, 28, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 320 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 321 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 322 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 323 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 324 | Tensor<[1, 28, 28, 256]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 325 | Tensor<[1, 28, 28, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 326 | Tensor<[1, 28, 28, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 327 | Tensor<[1, 28, 28, 256]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 328 | Tensor<[1, 28, 28, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 329 | Tensor<[1, 28, 56, 128]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 330 | Tensor<[1, 28, 56, 128]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 331 | Tensor<[1, 28, 56, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 332 | Tensor<[1, 28, 56, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 333 | Tensor<[1, 296, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                   | Done     | Done       | 1.0   | 0      |
| 334 | Tensor<[1, 296, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 44                                  | Done     | Done       | 1.0   | 0      |
| 335 | Tensor<[1, 296, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 168,<br>Optional[int] end = 296                                | Done     | Done       | 1.0   | 0      |
| 336 | Tensor<[1, 296, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 44,<br>Optional[int] end = 90                                  | Done     | Done       | 1.0   | 0      |
| 337 | Tensor<[1, 296, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 90,<br>Optional[int] end = 168                                 | Done     | Done       | 1.0   | 0      |
| 338 | Tensor<[1, 3, 224, 224]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 339 | Tensor<[1, 30, 40]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 340 | Tensor<[1, 30, 40]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 341 | Tensor<[1, 304, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 342 | Tensor<[1, 304, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 108,<br>Optional[int] end = 304                                | Done     | Done       | 1.0   | 0      |
| 343 | Tensor<[1, 304, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 108                                 | Done     | Done       | 1.0   | 0      |
| 344 | Tensor<[1, 310, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 345 | Tensor<[1, 310, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 54                                  | Done     | Done       | 1.0   | 0      |
| 346 | Tensor<[1, 310, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 54,<br>Optional[int] end = 310                                 | Done     | Done       | 1.0   | 0      |
| 347 | Tensor<[1, 32, 16, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 348 | Tensor<[1, 32, 32, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                   | Unknown  | Unknown    | N/A   | N/A    |
| 349 | Tensor<[1, 32, 32, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 350 | Tensor<[1, 32, 32, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 64,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
| 351 | Tensor<[1, 32, 32, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 352 | Tensor<[1, 32, 32, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 353 | Tensor<[1, 32, 32, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 354 | Tensor<[1, 32, 32, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 355 | Tensor<[1, 32, 32, 256]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 356 | Tensor<[1, 32, 32, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 357 | Tensor<[1, 32, 32, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 358 | Tensor<[1, 32, 32, 256]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 359 | Tensor<[1, 32, 32, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 360 | Tensor<[1, 32, 64, 128]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 361 | Tensor<[1, 32, 64, 128]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 362 | Tensor<[1, 32, 64, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 363 | Tensor<[1, 32, 64, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 364 | Tensor<[1, 320]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | -1     |
| 365 | Tensor<[1, 320]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 160                                          | Unknown  | Done       | 1.0   | 0      |
| 366 | Tensor<[1, 320]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | -1     |
| 367 | Tensor<[1, 320]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | 0      |
| 368 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 369 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 100,<br>Optional[int] end = 120                                | Done     | Done       | 1.0   | 0      |
| 370 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 120,<br>Optional[int] end = 140                                | Done     | Done       | 1.0   | 0      |
| 371 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 140,<br>Optional[int] end = 160                                | Done     | Done       | 1.0   | 0      |
| 372 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 328                                | Done     | Done       | 1.0   | 0      |
| 373 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 40                                  | Done     | Done       | 1.0   | 0      |
| 374 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 60                                  | Done     | Done       | 1.0   | 0      |
| 375 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 60,<br>Optional[int] end = 80                                  | Done     | Done       | 1.0   | 0      |
| 376 | Tensor<[1, 328, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 80,<br>Optional[int] end = 100                                 | Done     | Done       | 1.0   | 0      |
| 377 | Tensor<[1, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Removed  | Done       | 1.0   | -1     |
| 378 | Tensor<[1, 360, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 379 | Tensor<[1, 360, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 360                                 | Done     | Done       | 1.0   | 0      |
| 380 | Tensor<[1, 368, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 381 | Tensor<[1, 368, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 112,<br>Optional[int] end = 368                                | Done     | Done       | 1.0   | 0      |
| 382 | Tensor<[1, 368, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 54                                  | Done     | Done       | 1.0   | 0      |
| 383 | Tensor<[1, 368, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 54,<br>Optional[int] end = 112                                 | Done     | Done       | 1.0   | 0      |
| 384 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 192                                  | Done     | Done       | 1.0   | 0      |
| 385 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 96                                   | Done     | Done       | 1.0   | 0      |
| 386 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 192,<br>Optional[int] end = 288                                | Done     | Done       | 1.0   | 0      |
| 387 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 192,<br>Optional[int] end = 384                                | Done     | Done       | 1.0   | 0      |
| 388 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 288,<br>Optional[int] end = 384                                | Done     | Done       | 1.0   | 0      |
| 389 | Tensor<[1, 384, 35, 35]> self = ?,<br>int dim = 1,<br>Optional[int] start = 96,<br>Optional[int] end = 192                                 | Done     | Done       | 1.0   | 0      |
| 390 | Tensor<[1, 384, 512]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 391 | Tensor<[1, 40, 28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 392 | Tensor<[1, 40, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                    | Done     | Done       | 1.0   | 0      |
| 393 | Tensor<[1, 40, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 394 | Tensor<[1, 40, 28, 28]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 395 | Tensor<[1, 40, 28, 28]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 396 | Tensor<[1, 40]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 397 | Tensor<[1, 40]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 8                                             | Unknown  | Done       | 1.0   | 0      |
| 398 | Tensor<[1, 4150, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 399 | Tensor<[1, 4251, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 400 | Tensor<[1, 4251, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = -100,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | 0      |
| 401 | Tensor<[1, 4251, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = -100                                   | Removed  | Done       | 1.0   | 0      |
| 402 | Tensor<[1, 428, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 403 | Tensor<[1, 428, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 108,<br>Optional[int] end = 428                                | Done     | Done       | 1.0   | 0      |
| 404 | Tensor<[1, 428, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 108                                 | Done     | Done       | 1.0   | 0      |
| 405 | Tensor<[1, 448, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 128                                  | Done     | Done       | 1.0   | 0      |
| 406 | Tensor<[1, 448, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 128,<br>Optional[int] end = 256                                | Done     | Done       | 1.0   | 0      |
| 407 | Tensor<[1, 448, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 320                                | Done     | Done       | 1.0   | 0      |
| 408 | Tensor<[1, 448, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 320,<br>Optional[int] end = 448                                | Done     | Done       | 1.0   | 0      |
| 409 | Tensor<[1, 448, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 64                                   | Done     | Done       | 1.0   | 0      |
| 410 | Tensor<[1, 448, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 192,<br>Optional[int] end = 320                                | Done     | Done       | 1.0   | 0      |
| 411 | Tensor<[1, 448, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 320,<br>Optional[int] end = 448                                | Done     | Done       | 1.0   | 0      |
| 412 | Tensor<[1, 448, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 64,<br>Optional[int] end = 192                                 | Done     | Done       | 1.0   | 0      |
| 413 | Tensor<[1, 45]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 414 | Tensor<[1, 466, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                   | Done     | Done       | 1.0   | 0      |
| 415 | Tensor<[1, 466, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 112,<br>Optional[int] end = 210                                | Done     | Done       | 1.0   | 0      |
| 416 | Tensor<[1, 466, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 54                                  | Done     | Done       | 1.0   | 0      |
| 417 | Tensor<[1, 466, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 210,<br>Optional[int] end = 466                                | Done     | Done       | 1.0   | 0      |
| 418 | Tensor<[1, 466, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 54,<br>Optional[int] end = 112                                 | Done     | Done       | 1.0   | 0      |
| 419 | Tensor<[1, 46]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 420 | Tensor<[1, 46]> self = ?,<br>int dim = 1,<br>Optional[int] start = 45,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | 0      |
| 421 | Tensor<[1, 48, 112, 112]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 422 | Tensor<[1, 48, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 24                                  | Done     | Done       | 1.0   | 0      |
| 423 | Tensor<[1, 48, 112, 112]> self = ?,<br>int dim = 1,<br>Optional[int] start = 24,<br>Optional[int] end = 48                                 | Done     | Done       | 1.0   | 0      |
| 424 | Tensor<[1, 48, 112, 112]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 425 | Tensor<[1, 48, 112, 112]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Removed  | Done       | 1.0   | -1     |
| 426 | Tensor<[1, 480, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 427 | Tensor<[1, 480, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 240                                  | Done     | Done       | 1.0   | 0      |
| 428 | Tensor<[1, 480, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 240,<br>Optional[int] end = 480                                | Done     | Done       | 1.0   | 0      |
| 429 | Tensor<[1, 480, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 430 | Tensor<[1, 480, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 431 | Tensor<[1, 5, 1, 16]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 432 | Tensor<[1, 5, 16, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 433 | Tensor<[1, 5, 16, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 434 | Tensor<[1, 5, 16, 32]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 435 | Tensor<[1, 5, 16, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2   | Unknown  | Done       | 1.0   | 0      |
| 436 | Tensor<[1, 5, 16, 32]> self = ?,<br>int dim = 3,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2   | Unknown  | Done       | 1.0   | 0      |
| 437 | Tensor<[1, 5, 16, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 438 | Tensor<[1, 5, 16, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 439 | Tensor<[1, 5, 16, 64]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 440 | Tensor<[1, 5, 16, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                     | Unknown  | Done       | 1.0   | 0      |
| 441 | Tensor<[1, 5, 16, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 32,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
| 442 | Tensor<[1, 5, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | -1     |
| 443 | Tensor<[1, 5, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Done       | 1.0   | -1     |
| 444 | Tensor<[1, 50, 1024]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Removed  | Done       | 1.0   | -1     |
| 445 | Tensor<[1, 50, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
| 446 | Tensor<[1, 50, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 1                                        | Done     | Done       | 1.0   | 0      |
| 447 | Tensor<[1, 50, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 50                                       | Done     | Done       | 1.0   | 0      |
| 448 | Tensor<[1, 512, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                  | Done     | Done       | 1.0   | 0      |
| 449 | Tensor<[1, 512, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                | Done     | Done       | 1.0   | 0      |
| 450 | Tensor<[1, 512, 32, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                  | Done     | Done       | 1.0   | 0      |
| 451 | Tensor<[1, 512, 32, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                | Done     | Done       | 1.0   | 0      |
| 452 | Tensor<[1, 512, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                  | Done     | Done       | 1.0   | 0      |
| 453 | Tensor<[1, 512, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                | Done     | Done       | 1.0   | 0      |
| 454 | Tensor<[1, 512, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                    | Done     | Done       | 1.0   | 0      |
| 455 | Tensor<[1, 512, 8, 8]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                  | Done     | Done       | 1.0   | 0      |
| 456 | Tensor<[1, 512]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Done       | 1.0   | -1     |
| 457 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 12                                           | Removed  | Done       | 1.0   | 0      |
| 458 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                           | Removed  | Done       | 1.0   | 0      |
| 459 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                           | Removed  | Done       | 1.0   | 0      |
| 460 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 25                                           | Removed  | Done       | 1.0   | 0      |
| 461 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 384                                          | Removed  | Unknown    | N/A   | N/A    |
| 462 | Tensor<[1, 512]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9                                            | Done     | Done       | 1.0   | 0      |
| 463 | Tensor<[1, 514]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Done       | 1.0   | -1     |
| 464 | Tensor<[1, 514]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 10                                           | Done     | Done       | 1.0   | 0      |
| 465 | Tensor<[1, 51865]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                        | Unknown  | Unknown    | N/A   | N/A    |
| 466 | Tensor<[1, 54, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                    | Done     | Done       | 1.0   | 0      |
| 467 | Tensor<[1, 54, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 14,<br>Optional[int] end = 54                                   | Done     | Done       | 1.0   | 0      |
| 468 | Tensor<[1, 544, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 469 | Tensor<[1, 544, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 108,<br>Optional[int] end = 224                                | Done     | Done       | 1.0   | 0      |
| 470 | Tensor<[1, 544, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 224,<br>Optional[int] end = 544                                | Done     | Done       | 1.0   | 0      |
| 471 | Tensor<[1, 544, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 108                                 | Done     | Done       | 1.0   | 0      |
| 472 | Tensor<[1, 56, 56, 128]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 473 | Tensor<[1, 56, 56, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 474 | Tensor<[1, 56, 56, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 475 | Tensor<[1, 56, 56, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 476 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 477 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 478 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 479 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 480 | Tensor<[1, 59]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 481 | Tensor<[1, 59]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 482 | Tensor<[1, 5]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Unknown  | Done       | 1.0   | -1     |
| 483 | Tensor<[1, 6, 1, 15]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 484 | Tensor<[1, 6, 1, 17]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 485 | Tensor<[1, 6, 1, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 486 | Tensor<[1, 6, 1, s0 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
| 487 | Tensor<[1, 6, 17, 17]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 488 | Tensor<[1, 6, 17, 17]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 489 | Tensor<[1, 6, 17, 17]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
| 490 | Tensor<[1, 6, 18, 15]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 491 | Tensor<[1, 6, 18, 15]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | -1     |
| 492 | Tensor<[1, 6, 18, 15]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Done       | 1.0   | 0      |
| 493 | Tensor<[1, 6, 2, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 494 | Tensor<[1, 6, 2, 2]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 495 | Tensor<[1, 6, 2, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | 0      |
| 496 | Tensor<[1, 6, 3, 15]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 497 | Tensor<[1, 6, 3, 15]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 498 | Tensor<[1, 6, 3, 15]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | 0      |
| 499 | Tensor<[1, 6, s0 + 1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807            | Unknown  | Unknown    | N/A   | N/A    |
| 500 | Tensor<[1, 6, s0 + 1, s0 + 1]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807            | Unknown  | Unknown    | N/A   | N/A    |
| 501 | Tensor<[1, 6, s0 + 1, s0 + 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 502 | Tensor<[1, 6, s0 + 2, 15]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 503 | Tensor<[1, 6, s0 + 2, 15]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 504 | Tensor<[1, 6, s0 + 2, 15]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 505 | Tensor<[1, 60, 80]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 506 | Tensor<[1, 60, 80]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 507 | Tensor<[1, 60]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 508 | Tensor<[1, 60]> self = ?,<br>int dim = 1,<br>Optional[int] start = 59,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | 0      |
| 509 | Tensor<[1, 62, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                    | Done     | Done       | 1.0   | 0      |
| 510 | Tensor<[1, 62, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 62                                   | Done     | Done       | 1.0   | 0      |
| 511 | Tensor<[1, 64, 256, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                  | Done     | Done       | 1.0   | 0      |
| 512 | Tensor<[1, 64, 256, 256]> self = ?,<br>int dim = 1,<br>Optional[int] start = 32,<br>Optional[int] end = 64                                 | Done     | Done       | 1.0   | 0      |
| 513 | Tensor<[1, 64, 64, 128]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 514 | Tensor<[1, 64, 64, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 515 | Tensor<[1, 64, 64, 128]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 516 | Tensor<[1, 64, 64, 128]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 517 | Tensor<[1, 64, 64, 96]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 518 | Tensor<[1, 64, 64, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 519 | Tensor<[1, 64, 64, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 520 | Tensor<[1, 64, 64, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 521 | Tensor<[1, 640]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | -1     |
| 522 | Tensor<[1, 640]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Done       | 1.0   | -1     |
| 523 | Tensor<[1, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Done       | 1.0   | -1     |
| 524 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 525 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 120,<br>Optional[int] end = 160                                | Done     | Done       | 1.0   | 0      |
| 526 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 200                                | Done     | Done       | 1.0   | 0      |
| 527 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 200,<br>Optional[int] end = 240                                | Done     | Done       | 1.0   | 0      |
| 528 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 240,<br>Optional[int] end = 280                                | Done     | Done       | 1.0   | 0      |
| 529 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 280,<br>Optional[int] end = 320                                | Done     | Done       | 1.0   | 0      |
| 530 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 320,<br>Optional[int] end = 654                                | Done     | Done       | 1.0   | 0      |
| 531 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 80                                  | Done     | Done       | 1.0   | 0      |
| 532 | Tensor<[1, 654, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 80,<br>Optional[int] end = 120                                 | Done     | Done       | 1.0   | 0      |
| 533 | Tensor<[1, 672, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 534 | Tensor<[1, 672, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 336                                  | Done     | Done       | 1.0   | 0      |
| 535 | Tensor<[1, 672, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 336,<br>Optional[int] end = 672                                | Done     | Done       | 1.0   | 0      |
| 536 | Tensor<[1, 672, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 537 | Tensor<[1, 672, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 538 | Tensor<[1, 6]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Unknown  | Done       | 1.0   | -1     |
| 539 | Tensor<[1, 6]> self = ?,<br>int dim = 1,<br>Optional[int] start = 5,<br>Optional[int] end = 9223372036854775807                            | Unknown  | Done       | 1.0   | 0      |
| 540 | Tensor<[1, 7, 14, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 541 | Tensor<[1, 7, 14, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 542 | Tensor<[1, 7, 14, 512]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 543 | Tensor<[1, 7, 14, 512]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 544 | Tensor<[1, 7, 7, 1024]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 545 | Tensor<[1, 7, 7, 1024]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 546 | Tensor<[1, 7, 7, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 547 | Tensor<[1, 7, 7, 512]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 548 | Tensor<[1, 7, 7, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 549 | Tensor<[1, 7, 7, 768]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 550 | Tensor<[1, 7, 71, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Unknown    | N/A   | N/A    |
| 551 | Tensor<[1, 7, 73, 64]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = -2                                     | Unknown  | Unknown    | N/A   | N/A    |
| 552 | Tensor<[1, 7, 73, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Unknown    | N/A   | N/A    |
| 553 | Tensor<[1, 71, 7, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 32                                     | Unknown  | Unknown    | N/A   | N/A    |
| 554 | Tensor<[1, 71, 7, 64]> self = ?,<br>int dim = 3,<br>Optional[int] start = 32,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Unknown    | N/A   | N/A    |
| 555 | Tensor<[1, 72, 56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 556 | Tensor<[1, 72, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 36                                    | Done     | Done       | 1.0   | 0      |
| 557 | Tensor<[1, 72, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 36,<br>Optional[int] end = 72                                   | Done     | Done       | 1.0   | 0      |
| 558 | Tensor<[1, 72, 56, 56]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 559 | Tensor<[1, 72, 56, 56]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 560 | Tensor<[1, 736, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                  | Done     | Done       | 1.0   | 0      |
| 561 | Tensor<[1, 736, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 416                                | Done     | Done       | 1.0   | 0      |
| 562 | Tensor<[1, 736, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 416,<br>Optional[int] end = 576                                | Done     | Done       | 1.0   | 0      |
| 563 | Tensor<[1, 736, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 576,<br>Optional[int] end = 736                                | Done     | Done       | 1.0   | 0      |
| 564 | Tensor<[1, 740, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                   | Done     | Done       | 1.0   | 0      |
| 565 | Tensor<[1, 740, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 108,<br>Optional[int] end = 224                                | Done     | Done       | 1.0   | 0      |
| 566 | Tensor<[1, 740, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 224,<br>Optional[int] end = 420                                | Done     | Done       | 1.0   | 0      |
| 567 | Tensor<[1, 740, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 108                                 | Done     | Done       | 1.0   | 0      |
| 568 | Tensor<[1, 740, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 420,<br>Optional[int] end = 740                                | Done     | Done       | 1.0   | 0      |
| 569 | Tensor<[1, 768]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Done       | 1.0   | -1     |
| 570 | Tensor<[1, 77]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Removed  | Done       | 1.0   | -1     |
| 571 | Tensor<[1, 77]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 7                                             | Removed  | Done       | 1.0   | 0      |
| 572 | Tensor<[1, 78, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 20                                    | Done     | Done       | 1.0   | 0      |
| 573 | Tensor<[1, 78, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 20,<br>Optional[int] end = 78                                   | Done     | Done       | 1.0   | 0      |
| 574 | Tensor<[1, 78, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 14                                    | Done     | Done       | 1.0   | 0      |
| 575 | Tensor<[1, 78, 56, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 14,<br>Optional[int] end = 78                                   | Done     | Done       | 1.0   | 0      |
| 576 | Tensor<[1, 782, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 160                                    | Done     | Done       | 1.0   | 0      |
| 577 | Tensor<[1, 782, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 320                                  | Done     | Done       | 1.0   | 0      |
| 578 | Tensor<[1, 782, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 320,<br>Optional[int] end = 782                                  | Done     | Done       | 1.0   | 0      |
| 579 | Tensor<[1, 7]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Removed  | Done       | 1.0   | -1     |
| 580 | Tensor<[1, 8, 1, 10]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 581 | Tensor<[1, 8, 1, 2]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 582 | Tensor<[1, 8, 1, s0 + 1]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
| 583 | Tensor<[1, 8, 16, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 584 | Tensor<[1, 8, 16, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 585 | Tensor<[1, 8, 16, 512]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 586 | Tensor<[1, 8, 16, 512]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 587 | Tensor<[1, 8, 2, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 588 | Tensor<[1, 8, 2, 2]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Done       | 1.0   | -1     |
| 589 | Tensor<[1, 8, 2, 2]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | 0      |
| 590 | Tensor<[1, 8, 3, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 591 | Tensor<[1, 8, 3, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 592 | Tensor<[1, 8, 3, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                    | Unknown  | Done       | 1.0   | 0      |
| 593 | Tensor<[1, 8, 8, 1024]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 594 | Tensor<[1, 8, 8, 1024]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 595 | Tensor<[1, 8, 8, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 596 | Tensor<[1, 8, 8, 512]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 597 | Tensor<[1, 8, 8, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 598 | Tensor<[1, 8, 8, 768]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 599 | Tensor<[1, 8, s0 + 1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807            | Unknown  | Unknown    | N/A   | N/A    |
| 600 | Tensor<[1, 8, s0 + 1, s0 + 1]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807            | Unknown  | Unknown    | N/A   | N/A    |
| 601 | Tensor<[1, 8, s0 + 1, s0 + 1]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807           | Unknown  | Unknown    | N/A   | N/A    |
| 602 | Tensor<[1, 8, s0 + 2, 10]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 603 | Tensor<[1, 8, s0 + 2, 10]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                | Unknown  | Unknown    | N/A   | N/A    |
| 604 | Tensor<[1, 8, s0 + 2, 10]> self = ?,<br>int dim = 2,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807               | Unknown  | Unknown    | N/A   | N/A    |
| 605 | Tensor<[1, 80, 14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 606 | Tensor<[1, 80, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 40                                    | Done     | Done       | 1.0   | 0      |
| 607 | Tensor<[1, 80, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 40,<br>Optional[int] end = 80                                   | Done     | Done       | 1.0   | 0      |
| 608 | Tensor<[1, 80, 14, 14]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 609 | Tensor<[1, 80, 14, 14]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 610 | Tensor<[1, 80, 3000]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 611 | Tensor<[1, 80, 3000]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                     | Unknown  | Done       | 1.0   | -1     |
| 612 | Tensor<[1, 800, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 160                                    | Done     | Done       | 1.0   | 0      |
| 613 | Tensor<[1, 800, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 160,<br>Optional[int] end = 800                                  | Done     | Done       | 1.0   | 0      |
| 614 | Tensor<[1, 896, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 256                                  | Done     | Done       | 1.0   | 0      |
| 615 | Tensor<[1, 896, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                | Done     | Done       | 1.0   | 0      |
| 616 | Tensor<[1, 896, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 512,<br>Optional[int] end = 640                                | Done     | Done       | 1.0   | 0      |
| 617 | Tensor<[1, 896, 14, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 640,<br>Optional[int] end = 896                                | Done     | Done       | 1.0   | 0      |
| 618 | Tensor<[1, 9, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Removed  | Done       | 1.0   | -1     |
| 619 | Tensor<[1, 94, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 16                                    | Done     | Done       | 1.0   | 0      |
| 620 | Tensor<[1, 94, 28, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 16,<br>Optional[int] end = 94                                   | Done     | Done       | 1.0   | 0      |
| 621 | Tensor<[1, 960, 7, 7]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 622 | Tensor<[1, 960, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 480                                    | Done     | Done       | 1.0   | 0      |
| 623 | Tensor<[1, 960, 7, 7]> self = ?,<br>int dim = 1,<br>Optional[int] start = 480,<br>Optional[int] end = 960                                  | Done     | Done       | 1.0   | 0      |
| 624 | Tensor<[1, 960, 7, 7]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 625 | Tensor<[1, 960, 7, 7]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 626 | Tensor<[1, s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Unknown  | Unknown    | N/A   | N/A    |
| 627 | Tensor<[1, s0 + 1]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Unknown  | Unknown    | N/A   | N/A    |
| 628 | Tensor<[1, s0]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                           | Unknown  | Unknown    | N/A   | N/A    |
| 629 | Tensor<[1, s0]> self = ?,<br>int dim = 1,<br>Optional[int] start = -1,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Unknown    | N/A   | N/A    |
| 630 | Tensor<[1, s1 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                       | Unknown  | Unknown    | N/A   | N/A    |
| 631 | Tensor<[1, s1 + 1]> self = ?,<br>int dim = 1,<br>Optional[int]<s1> start = ?,<br>Optional[int] end = 9223372036854775807                   | Unknown  | Unknown    | N/A   | N/A    |
| 632 | Tensor<[1, s10 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Unknown  | Unknown    | N/A   | N/A    |
| 633 | Tensor<[1, s10 + 1]> self = ?,<br>int dim = 1,<br>Optional[int]<s10> start = ?,<br>Optional[int] end = 9223372036854775807                 | Unknown  | Unknown    | N/A   | N/A    |
| 634 | Tensor<[1152]> self = ?,<br>int dim = 0,<br>Optional[int] start = 384,<br>Optional[int] end = 768                                          | Removed  | Unknown    | N/A   | N/A    |
| 635 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 636 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 637 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 638 | Tensor<[1536]> self = ?,<br>int dim = 0,<br>Optional[int] start = 512,<br>Optional[int] end = 1024                                         | Removed  | Unknown    | N/A   | N/A    |
| 639 | Tensor<[16, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 640 | Tensor<[16, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                          | Done     | Unknown    | N/A   | N/A    |
| 641 | Tensor<[16, 16]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                           | Done     | Unknown    | N/A   | N/A    |
| 642 | Tensor<[192, 2]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Unknown    | N/A   | N/A    |
| 643 | Tensor<[2, 1, 1, 7]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                      | Removed  | Done       | 1.0   | -1     |
| 644 | Tensor<[2, 7]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Removed  | Done       | 1.0   | -1     |
| 645 | Tensor<[2048, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 7                                          | Unknown  | Unknown    | N/A   | N/A    |
| 646 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 647 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 648 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Removed  | Unknown    | N/A   | N/A    |
| 649 | Tensor<[2304]> self = ?,<br>int dim = 0,<br>Optional[int] start = 768,<br>Optional[int] end = 1536                                         | Removed  | Unknown    | N/A   | N/A    |
| 650 | Tensor<[24, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 651 | Tensor<[24, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                          | Done     | Unknown    | N/A   | N/A    |
| 652 | Tensor<[24, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                           | Removed  | Unknown    | N/A   | N/A    |
| 653 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 654 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 655 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 656 | Tensor<[288]> self = ?,<br>int dim = 0,<br>Optional[int] start = 96,<br>Optional[int] end = 192                                            | Removed  | Unknown    | N/A   | N/A    |
| 657 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 658 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 659 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 660 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 661 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 662 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 663 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 664 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 665 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 666 | Tensor<[3072]> self = ?,<br>int dim = 0,<br>Optional[int] start = 1024,<br>Optional[int] end = 2048                                        | Removed  | Unknown    | N/A   | N/A    |
| 667 | Tensor<[32, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 668 | Tensor<[32, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                          | Done     | Unknown    | N/A   | N/A    |
| 669 | Tensor<[32, 32]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                           | Done     | Unknown    | N/A   | N/A    |
| 670 | Tensor<[3234, 4]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Removed  | Done       | 1.0   | -1     |
| 671 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 2                                           | Done     | Done       | 1.0   | 0      |
| 672 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2        | Done     | Unknown    | N/A   | N/A    |
| 673 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 674 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2        | Done     | Unknown    | N/A   | N/A    |
| 675 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 676 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 2,<br>Optional[int] end = 9223372036854775807                         | Done     | Done       | 1.0   | 0      |
| 677 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 2,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 678 | Tensor<[3234, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 3,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 679 | Tensor<[3234]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Removed  | Unknown    | N/A   | N/A    |
| 680 | Tensor<[384]> self = ?,<br>int dim = 0,<br>Optional[int] start = 128,<br>Optional[int] end = 256                                           | Removed  | Unknown    | N/A   | N/A    |
| 681 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 682 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 683 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 684 | Tensor<[4, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 685 | Tensor<[4, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                           | Done     | Unknown    | N/A   | N/A    |
| 686 | Tensor<[4, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                            | Done     | Unknown    | N/A   | N/A    |
| 687 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 688 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 689 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 690 | Tensor<[4, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 691 | Tensor<[4, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                           | Done     | Unknown    | N/A   | N/A    |
| 692 | Tensor<[4, 32]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                            | Done     | Unknown    | N/A   | N/A    |
| 693 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 694 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 695 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 696 | Tensor<[4, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 697 | Tensor<[4, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                           | Done     | Unknown    | N/A   | N/A    |
| 698 | Tensor<[4, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                            | Done     | Unknown    | N/A   | N/A    |
| 699 | Tensor<[448, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 1                                          | Unknown  | Done       | 1.0   | 0      |
| 700 | Tensor<[448, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 4                                          | Unknown  | Done       | 1.0   | 0      |
| 701 | Tensor<[448, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 4,<br>Optional[int] end = 5                                          | Unknown  | Done       | 1.0   | 0      |
| 702 | Tensor<[448, 768]> self = ?,<br>int dim = 0,<br>Optional[int]<s2> start = ?,<br>Optional[int]<s2 + 1> end = ?                              | Unknown  | Failed     | N/A   | N/A    |
| 703 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 704 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 705 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Removed  | Unknown    | N/A   | N/A    |
| 706 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 707 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 708 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 709 | Tensor<[56, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 710 | Tensor<[56, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                          | Done     | Unknown    | N/A   | N/A    |
| 711 | Tensor<[56, 64]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                           | Removed  | Unknown    | N/A   | N/A    |
| 712 | Tensor<[576]> self = ?,<br>int dim = 0,<br>Optional[int] start = 192,<br>Optional[int] end = 384                                           | Removed  | Unknown    | N/A   | N/A    |
| 713 | Tensor<[64, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 714 | Tensor<[64, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                          | Done     | Unknown    | N/A   | N/A    |
| 715 | Tensor<[64, 64]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                           | Done     | Unknown    | N/A   | N/A    |
| 716 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 717 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 718 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Removed  | Unknown    | N/A   | N/A    |
| 719 | Tensor<[768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 256,<br>Optional[int] end = 512                                           | Removed  | Unknown    | N/A   | N/A    |
| 720 | Tensor<[8, 1, 1, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Unknown    | N/A   | N/A    |
| 721 | Tensor<[8, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = -4,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 722 | Tensor<[8, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = -8,<br>Optional[int] end = -4                                           | Done     | Unknown    | N/A   | N/A    |
| 723 | Tensor<[8, 16]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -8                                            | Removed  | Unknown    | N/A   | N/A    |
| 724 | Tensor<[8, 384]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Removed  | Unknown    | N/A   | N/A    |
| 725 | Tensor<[851]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                             | Removed  | Unknown    | N/A   | N/A    |
| 726 | Tensor<[8732, 4]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                         | Removed  | Done       | 1.0   | -1     |
| 727 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 2                                           | Done     | Done       | 1.0   | 0      |
| 728 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2        | Done     | Unknown    | N/A   | N/A    |
| 729 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 730 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2        | Done     | Unknown    | N/A   | N/A    |
| 731 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 732 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 2,<br>Optional[int] end = 9223372036854775807                         | Done     | Done       | 1.0   | 0      |
| 733 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 2,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 734 | Tensor<[8732, 4]> self = ?,<br>int dim = 1,<br>Optional[int] start = 3,<br>Optional[int] end = 9223372036854775807,<br>int step = 4        | Done     | Unknown    | N/A   | N/A    |
| 735 | Tensor<[8732]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                            | Removed  | Unknown    | N/A   | N/A    |
| 736 | Tensor<[s0 + 1]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                          | Unknown  | Unknown    | N/A   | N/A    |

