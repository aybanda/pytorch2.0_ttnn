### aten.mm.default
|     | ATen Input Variations                                            | Status   | Isolated   | PCC                | Host   |
|----:|:-----------------------------------------------------------------|:---------|:-----------|:-------------------|:-------|
|   0 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 1024]> mat2 = ?     | Done     | Done       | 0.9999698981032241 | 0      |
|   1 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 1280]> mat2 = ?     | Done     | Done       | 0.9999697620640818 | 0      |
|   2 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 1536]> mat2 = ?     | Done     | Done       | 0.9999686171319188 | 0      |
|   3 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 2048]> mat2 = ?     | Done     | Done       | 0.9999691106986145 | 0      |
|   4 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 512]> mat2 = ?      | Done     | Done       | 0.9999652963952465 | 0      |
|   5 | Tensor<[1, 1000]> self = ?,<br>Tensor<[1000, 768]> mat2 = ?      | Done     | Done       | 0.9999673290268833 | 0      |
|   6 | Tensor<[1, 1024]> self = ?,<br>Tensor<[1024, 1024]> mat2 = ?     | Unknown  | Done       | 0.9999656488764666 | 0      |
|   7 | Tensor<[1, 1024]> self = ?,<br>Tensor<[1024, 3072]> mat2 = ?     | Unknown  | Done       | 0.9999643777923425 | 0      |
|   8 | Tensor<[1, 1024]> self = ?,<br>Tensor<[1024, 32128]> mat2 = ?    | Unknown  | Done       | 0.9999657318508302 | 0      |
|   9 | Tensor<[1, 1024]> self = ?,<br>Tensor<[1024, 4096]> mat2 = ?     | Unknown  | Done       | 0.9999658210511346 | 0      |
|  10 | Tensor<[1, 1024]> self = ?,<br>Tensor<[1024, 512]> mat2 = ?      | Unknown  | Done       | 0.9999626953541195 | 0      |
|  11 | Tensor<[1, 10]> self = ?,<br>Tensor<[10, 128]> mat2 = ?          | Done     | Done       | 0.9999932794228862 | 0      |
|  12 | Tensor<[1, 128]> self = ?,<br>Tensor<[128, 64]> mat2 = ?         | Done     | Done       | 0.9999780456453847 | 0      |
|  13 | Tensor<[1, 128]> self = ?,<br>Tensor<[128, 784]> mat2 = ?        | Done     | Done       | 0.9999794190166511 | 0      |
|  14 | Tensor<[1, 128]> self = ?,<br>Tensor<[128, 9216]> mat2 = ?       | Done     | Done       | 0.9999794140367623 | 0      |
|  15 | Tensor<[1, 12]> self = ?,<br>Tensor<[12, 3]> mat2 = ?            | Done     | Done       | 0.998906107238672  | 0      |
|  16 | Tensor<[1, 12]> self = ?,<br>Tensor<[12, 64]> mat2 = ?           | Done     | Done       | 0.9999891945802225 | 0      |
|  17 | Tensor<[1, 2048]> self = ?,<br>Tensor<[2048, 512]> mat2 = ?      | Unknown  | Done       | 0.999960728843302  | 0      |
|  18 | Tensor<[1, 21843]> self = ?,<br>Tensor<[21843, 768]> mat2 = ?    | Done     | Done       | 0.9995068309996874 | 0      |
|  19 | Tensor<[1, 2]> self = ?,<br>Tensor<[2, 512]> mat2 = ?            | Done     | Done       | 0.9999916541309198 | 0      |
|  20 | Tensor<[1, 3072]> self = ?,<br>Tensor<[3072, 768]> mat2 = ?      | Unknown  | Done       | 0.9999443520202527 | 0      |
|  21 | Tensor<[1, 384]> self = ?,<br>Tensor<[384, 512]> mat2 = ?        | Unknown  | Done       | 0.9999751239421292 | 0      |
|  22 | Tensor<[1, 3]> self = ?,<br>Tensor<[3, 12]> mat2 = ?             | Done     | Done       | 0.9999966045368508 | 0      |
|  23 | Tensor<[1, 4096]> self = ?,<br>Tensor<[4096, 1024]> mat2 = ?     | Unknown  | Done       | 0.9999358870237913 | 0      |
|  24 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 1024]> mat2 = ?       | Unknown  | Done       | 0.999970699862762  | 0      |
|  25 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 2048]> mat2 = ?       | Unknown  | Done       | 0.9999730782506949 | 0      |
|  26 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 32128]> mat2 = ?      | Unknown  | Done       | 0.9999712381426427 | 0      |
|  27 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 384]> mat2 = ?        | Unknown  | Done       | 0.999969681464183  | 0      |
|  28 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 50272]> mat2 = ?      | Unknown  | Done       | 0.9999721274476456 | 0      |
|  29 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 512]> mat2 = ?        | Unknown  | Done       | 0.9999743404523056 | 0      |
|  30 | Tensor<[1, 512]> self = ?,<br>Tensor<[512, 768]> mat2 = ?        | Done     | Done       | 0.9999711653502095 | 0      |
|  31 | Tensor<[1, 64]> self = ?,<br>Tensor<[64, 128]> mat2 = ?          | Done     | Done       | 0.999987538676787  | 0      |
|  32 | Tensor<[1, 64]> self = ?,<br>Tensor<[64, 12]> mat2 = ?           | Done     | Done       | 0.9999856774509178 | 0      |
|  33 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 3072]> mat2 = ?       | Unknown  | Done       | 0.9999686111280203 | 0      |
|  34 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 32128]> mat2 = ?      | Unknown  | Done       | 0.999968779361499  | 0      |
|  35 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 50257]> mat2 = ?      | Unknown  | Done       | 0.9999679565139534 | 0      |
|  36 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 512]> mat2 = ?        | Done     | Done       | 0.9999710263641663 | 0      |
|  37 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 51865]> mat2 = ?      | Unknown  | Done       | 0.9999687129303795 | 0      |
|  38 | Tensor<[1, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?        | Unknown  | Done       | 0.9999659791903754 | 0      |
|  39 | Tensor<[1, 784]> self = ?,<br>Tensor<[784, 128]> mat2 = ?        | Done     | Done       | 0.9999617484298913 | 0      |
|  40 | Tensor<[10, 1024]> self = ?,<br>Tensor<[1024, 1024]> mat2 = ?    | Unknown  | Done       | 0.9999637391025249 | 0      |
|  41 | Tensor<[10, 1024]> self = ?,<br>Tensor<[1024, 4096]> mat2 = ?    | Unknown  | Done       | 0.9999657462261405 | 0      |
|  42 | Tensor<[10, 1]> self = ?,<br>Tensor<[1, 128]> mat2 = ?           | Done     | Done       | 0.9999943649058725 | 0      |
|  43 | Tensor<[10, 2048]> self = ?,<br>Tensor<[2048, 512]> mat2 = ?     | Unknown  | Done       | 0.9999527853071405 | 0      |
|  44 | Tensor<[10, 3072]> self = ?,<br>Tensor<[3072, 768]> mat2 = ?     | Unknown  | Done       | 0.9999442714932767 | 0      |
|  45 | Tensor<[10, 4096]> self = ?,<br>Tensor<[4096, 1024]> mat2 = ?    | Unknown  | Done       | 0.9999336197445432 | 0      |
|  46 | Tensor<[10, 512]> self = ?,<br>Tensor<[512, 2048]> mat2 = ?      | Unknown  | Done       | 0.9999713902765629 | 0      |
|  47 | Tensor<[10, 512]> self = ?,<br>Tensor<[512, 512]> mat2 = ?       | Unknown  | Done       | 0.999971321870713  | 0      |
|  48 | Tensor<[10, 768]> self = ?,<br>Tensor<[768, 3072]> mat2 = ?      | Unknown  | Done       | 0.9999685295568667 | 0      |
|  49 | Tensor<[10, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?       | Unknown  | Done       | 0.9999681870020068 | 0      |
|  50 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 1024]> mat2 = ?        | Done     | Done       | 0.9999921984274246 | 0      |
|  51 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 1280]> mat2 = ?        | Done     | Done       | 0.9999920987431303 | 0      |
|  52 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 1536]> mat2 = ?        | Done     | Done       | 0.9999923102286085 | 0      |
|  53 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 2048]> mat2 = ?        | Done     | Done       | 0.9999922362791959 | 0      |
|  54 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 512]> mat2 = ?         | Done     | Done       | 0.9999923242539251 | 0      |
|  55 | Tensor<[1000, 1]> self = ?,<br>Tensor<[1, 768]> mat2 = ?         | Done     | Done       | 0.999991975860218  | 0      |
|  56 | Tensor<[1024, 160]> self = ?,<br>Tensor<[160, 160]> mat2 = ?     | Done     | Done       | 0.9999820697598679 | 0      |
|  57 | Tensor<[1024, 160]> self = ?,<br>Tensor<[160, 256]> mat2 = ?     | Done     | Done       | 0.9999820279152958 | 0      |
|  58 | Tensor<[1024, 160]> self = ?,<br>Tensor<[160, 640]> mat2 = ?     | Done     | Done       | 0.9999819094675215 | 0      |
|  59 | Tensor<[1024, 256]> self = ?,<br>Tensor<[256, 256]> mat2 = ?     | Done     | Done       | 0.999975317878174  | 0      |
|  60 | Tensor<[1024, 384]> self = ?,<br>Tensor<[384, 192]> mat2 = ?     | Done     | Done       | 0.9999730680386595 | 0      |
|  61 | Tensor<[1024, 512]> self = ?,<br>Tensor<[512, 256]> mat2 = ?     | Done     | Done       | 0.9999711951576697 | 0      |
|  62 | Tensor<[1024, 640]> self = ?,<br>Tensor<[640, 160]> mat2 = ?     | Done     | Done       | 0.9999698267737818 | 0      |
|  63 | Tensor<[12, 1]> self = ?,<br>Tensor<[1, 3]> mat2 = ?             | Done     | Done       | 0.9999902954242642 | 0      |
|  64 | Tensor<[12, 1]> self = ?,<br>Tensor<[1, 64]> mat2 = ?            | Done     | Done       | 0.9999914441021333 | 0      |
|  65 | Tensor<[128, 16384]> self = ?,<br>Tensor<[16384, 32]> mat2 = ?   | Done     | Done       | 0.9998063673772125 | 0      |
|  66 | Tensor<[128, 1]> self = ?,<br>Tensor<[1, 64]> mat2 = ?           | Done     | Done       | 0.9999920627373061 | 0      |
|  67 | Tensor<[128, 1]> self = ?,<br>Tensor<[1, 784]> mat2 = ?          | Done     | Done       | 0.9999927709588773 | 0      |
|  68 | Tensor<[128, 1]> self = ?,<br>Tensor<[1, 9216]> mat2 = ?         | Done     | Done       | 0.9999927715739475 | 0      |
|  69 | Tensor<[14, 2048]> self = ?,<br>Tensor<[2048, 512]> mat2 = ?     | Done     | Done       | 0.9999559834403867 | 0      |
|  70 | Tensor<[14, 512]> self = ?,<br>Tensor<[512, 2048]> mat2 = ?      | Done     | Done       | 0.9999716402738676 | 0      |
|  71 | Tensor<[14, 512]> self = ?,<br>Tensor<[512, 512]> mat2 = ?       | Done     | Done       | 0.9999714605669697 | 0      |
|  72 | Tensor<[15, 1024]> self = ?,<br>Tensor<[1024, 512]> mat2 = ?     | Unknown  | Done       | 0.9999649492531962 | 0      |
|  73 | Tensor<[15, 384]> self = ?,<br>Tensor<[384, 512]> mat2 = ?       | Unknown  | Done       | 0.9999724241868082 | 0      |
|  74 | Tensor<[15, 512]> self = ?,<br>Tensor<[512, 1024]> mat2 = ?      | Unknown  | Done       | 0.999970809743136  | 0      |
|  75 | Tensor<[15, 512]> self = ?,<br>Tensor<[512, 384]> mat2 = ?       | Unknown  | Done       | 0.9999716286541644 | 0      |
|  76 | Tensor<[1500, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?     | Unknown  | Done       | 0.9999633127461951 | 0      |
|  77 | Tensor<[160, 1024]> self = ?,<br>Tensor<[1024, 160]> mat2 = ?    | Done     | Done       | 0.9999552048119282 | 0      |
|  78 | Tensor<[160, 1024]> self = ?,<br>Tensor<[1024, 640]> mat2 = ?    | Done     | Done       | 0.999965400271444  | 0      |
|  79 | Tensor<[160, 256]> self = ?,<br>Tensor<[256, 1024]> mat2 = ?     | Done     | Done       | 0.9999753991183321 | 0      |
|  80 | Tensor<[160, 256]> self = ?,<br>Tensor<[256, 160]> mat2 = ?      | Done     | Done       | 0.999979415066273  | 0      |
|  81 | Tensor<[16384, 128]> self = ?,<br>Tensor<[128, 32]> mat2 = ?     | Done     | Done       | 0.9999798655374363 | 0      |
|  82 | Tensor<[16384, 32]> self = ?,<br>Tensor<[32, 128]> mat2 = ?      | Done     | Done       | 0.9999895157742837 | 0      |
|  83 | Tensor<[16384, 32]> self = ?,<br>Tensor<[32, 256]> mat2 = ?      | Done     | Done       | 0.999989455840206  | 0      |
|  84 | Tensor<[16384, 32]> self = ?,<br>Tensor<[32, 32]> mat2 = ?       | Done     | Done       | 0.9999895099552555 | 0      |
|  85 | Tensor<[196, 1024]> self = ?,<br>Tensor<[1024, 512]> mat2 = ?    | Done     | Done       | 0.9999551768375411 | 0      |
|  86 | Tensor<[196, 3072]> self = ?,<br>Tensor<[3072, 768]> mat2 = ?    | Done     | Done       | 0.9999448241897535 | 0      |
|  87 | Tensor<[196, 384]> self = ?,<br>Tensor<[384, 768]> mat2 = ?      | Done     | Done       | 0.9999731178437185 | 0      |
|  88 | Tensor<[196, 768]> self = ?,<br>Tensor<[768, 3072]> mat2 = ?     | Done     | Done       | 0.9999682927271227 | 0      |
|  89 | Tensor<[196, 768]> self = ?,<br>Tensor<[768, 384]> mat2 = ?      | Done     | Done       | 0.9999636500418981 | 0      |
|  90 | Tensor<[197, 3072]> self = ?,<br>Tensor<[3072, 768]> mat2 = ?    | Done     | Done       | 0.9999444896865558 | 0      |
|  91 | Tensor<[197, 768]> self = ?,<br>Tensor<[768, 3072]> mat2 = ?     | Done     | Done       | 0.999968412677724  | 0      |
|  92 | Tensor<[197, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?      | Done     | Done       | 0.9999682484729605 | 0      |
|  93 | Tensor<[2, 1]> self = ?,<br>Tensor<[1, 512]> mat2 = ?            | Done     | Done       | 0.9999978570586816 | 0      |
|  94 | Tensor<[2, 512]> self = ?,<br>Tensor<[512, 1]> mat2 = ?          | Done     | Done       | 1.0                | 0      |
|  95 | Tensor<[2, 512]> self = ?,<br>Tensor<[512, 512]> mat2 = ?        | Done     | Done       | 0.9999696831422862 | 0      |
|  96 | Tensor<[2048, 14]> self = ?,<br>Tensor<[14, 512]> mat2 = ?       | Done     | Done       | 0.999991491018755  | 0      |
|  97 | Tensor<[2048, 768]> self = ?,<br>Tensor<[768, 262]> mat2 = ?     | Removed  | Done       | 0.9999683331233449 | 0      |
|  98 | Tensor<[21843, 1]> self = ?,<br>Tensor<[1, 768]> mat2 = ?        | Done     | Done       | 0.999992182354889  | 0      |
|  99 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 12]> mat2 = ?       | Removed  | Done       | 0.9999733020142749 | 0      |
| 100 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 16]> mat2 = ?       | Removed  | Done       | 0.9999700835464506 | 0      |
| 101 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 24]> mat2 = ?       | Removed  | Done       | 0.9999718345182341 | 0      |
| 102 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 32]> mat2 = ?       | Removed  | Done       | 0.9999716997617811 | 0      |
| 103 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 3]> mat2 = ?        | Removed  | Done       | 0.9999696033573018 | 0      |
| 104 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 4]> mat2 = ?        | Removed  | Done       | 0.9999718057981928 | 0      |
| 105 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 6]> mat2 = ?        | Removed  | Done       | 0.9999690964052599 | 0      |
| 106 | Tensor<[225, 512]> self = ?,<br>Tensor<[512, 8]> mat2 = ?        | Removed  | Done       | 0.9999692185529421 | 0      |
| 107 | Tensor<[256, 1024]> self = ?,<br>Tensor<[1024, 160]> mat2 = ?    | Done     | Done       | 0.9999554292664561 | 0      |
| 108 | Tensor<[256, 1024]> self = ?,<br>Tensor<[1024, 256]> mat2 = ?    | Done     | Done       | 0.9999548261510073 | 0      |
| 109 | Tensor<[256, 1024]> self = ?,<br>Tensor<[1024, 512]> mat2 = ?    | Done     | Done       | 0.9999553021886753 | 0      |
| 110 | Tensor<[256, 160]> self = ?,<br>Tensor<[160, 160]> mat2 = ?      | Done     | Done       | 0.9999821294933365 | 0      |
| 111 | Tensor<[256, 16384]> self = ?,<br>Tensor<[16384, 32]> mat2 = ?   | Done     | Done       | 0.9998028777146902 | 0      |
| 112 | Tensor<[256, 256]> self = ?,<br>Tensor<[256, 1024]> mat2 = ?     | Done     | Done       | 0.9999753738032722 | 0      |
| 113 | Tensor<[256, 256]> self = ?,<br>Tensor<[256, 256]> mat2 = ?      | Done     | Done       | 0.9999794198451551 | 0      |
| 114 | Tensor<[256, 256]> self = ?,<br>Tensor<[256, 512]> mat2 = ?      | Done     | Done       | 0.9999791470252442 | 0      |
| 115 | Tensor<[256, 32]> self = ?,<br>Tensor<[32, 32]> mat2 = ?         | Done     | Done       | 0.9999892830879694 | 0      |
| 116 | Tensor<[256, 4096]> self = ?,<br>Tensor<[4096, 64]> mat2 = ?     | Done     | Done       | 0.9998468115457974 | 0      |
| 117 | Tensor<[256, 512]> self = ?,<br>Tensor<[512, 256]> mat2 = ?      | Done     | Done       | 0.9999713798072445 | 0      |
| 118 | Tensor<[256, 512]> self = ?,<br>Tensor<[512, 768]> mat2 = ?      | Done     | Done       | 0.9999711815051172 | 0      |
| 119 | Tensor<[256, 64]> self = ?,<br>Tensor<[64, 64]> mat2 = ?         | Done     | Done       | 0.9999865901785695 | 0      |
| 120 | Tensor<[256, 768]> self = ?,<br>Tensor<[768, 384]> mat2 = ?      | Done     | Done       | 0.9999633976096383 | 0      |
| 121 | Tensor<[3, 1]> self = ?,<br>Tensor<[1, 12]> mat2 = ?             | Done     | Done       | 0.9999970095370196 | 0      |
| 122 | Tensor<[3072, 196]> self = ?,<br>Tensor<[196, 768]> mat2 = ?     | Done     | Done       | 0.9999821036076768 | 0      |
| 123 | Tensor<[3072, 197]> self = ?,<br>Tensor<[197, 768]> mat2 = ?     | Done     | Done       | 0.9999820280141292 | 0      |
| 124 | Tensor<[3072, 50]> self = ?,<br>Tensor<[50, 768]> mat2 = ?       | Done     | Done       | 0.9999859790956943 | 0      |
| 125 | Tensor<[32, 11008]> self = ?,<br>Tensor<[11008, 4096]> mat2 = ?  | Unknown  | Unknown    | N/A                | N/A    |
| 126 | Tensor<[32, 1536]> self = ?,<br>Tensor<[1536, 250880]> mat2 = ?  | Done     | Done       | 0.9999602158774988 | 0      |
| 127 | Tensor<[32, 16384]> self = ?,<br>Tensor<[16384, 128]> mat2 = ?   | Done     | Done       | 0.9997990216193955 | 0      |
| 128 | Tensor<[32, 16384]> self = ?,<br>Tensor<[16384, 32]> mat2 = ?    | Done     | Done       | 0.9998176621572186 | 0      |
| 129 | Tensor<[32, 256]> self = ?,<br>Tensor<[256, 16384]> mat2 = ?     | Done     | Done       | 0.9999752937832103 | 0      |
| 130 | Tensor<[32, 256]> self = ?,<br>Tensor<[256, 32]> mat2 = ?        | Done     | Done       | 0.9999739456846726 | 0      |
| 131 | Tensor<[32, 4096]> self = ?,<br>Tensor<[4096, 11008]> mat2 = ?   | Unknown  | Unknown    | N/A                | N/A    |
| 132 | Tensor<[32, 4096]> self = ?,<br>Tensor<[4096, 32000]> mat2 = ?   | Unknown  | Unknown    | N/A                | N/A    |
| 133 | Tensor<[32, 4096]> self = ?,<br>Tensor<[4096, 4096]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |
| 134 | Tensor<[384, 768]> self = ?,<br>Tensor<[768, 196]> mat2 = ?      | Done     | Done       | 0.9999636713563053 | 0      |
| 135 | Tensor<[4, 768]> self = ?,<br>Tensor<[768, 51865]> mat2 = ?      | Unknown  | Done       | 0.9999683471386521 | 0      |
| 136 | Tensor<[4, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?        | Unknown  | Done       | 0.9999672867579807 | 0      |
| 137 | Tensor<[4096, 256]> self = ?,<br>Tensor<[256, 64]> mat2 = ?      | Done     | Done       | 0.9999752885419916 | 0      |
| 138 | Tensor<[4096, 320]> self = ?,<br>Tensor<[320, 320]> mat2 = ?     | Unknown  | Done       | 0.9999741553355532 | 0      |
| 139 | Tensor<[4096, 64]> self = ?,<br>Tensor<[64, 256]> mat2 = ?       | Done     | Done       | 0.9999863711722274 | 0      |
| 140 | Tensor<[4096, 64]> self = ?,<br>Tensor<[64, 64]> mat2 = ?        | Done     | Done       | 0.9999864415336909 | 0      |
| 141 | Tensor<[45, 768]> self = ?,<br>Tensor<[768, 50257]> mat2 = ?     | Unknown  | Done       | 0.9999682889509482 | 0      |
| 142 | Tensor<[45, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?       | Unknown  | Done       | 0.9999681430579546 | 0      |
| 143 | Tensor<[49, 1536]> self = ?,<br>Tensor<[1536, 768]> mat2 = ?     | Done     | Done       | 0.9999607631856176 | 0      |
| 144 | Tensor<[49, 2048]> self = ?,<br>Tensor<[2048, 1024]> mat2 = ?    | Done     | Done       | 0.9999547562671323 | 0      |
| 145 | Tensor<[5, 1024]> self = ?,<br>Tensor<[1024, 1024]> mat2 = ?     | Unknown  | Done       | 0.9999645058919459 | 0      |
| 146 | Tensor<[5, 1024]> self = ?,<br>Tensor<[1024, 3072]> mat2 = ?     | Unknown  | Done       | 0.9999655702692375 | 0      |
| 147 | Tensor<[50, 3072]> self = ?,<br>Tensor<[3072, 768]> mat2 = ?     | Done     | Done       | 0.9999449422767561 | 0      |
| 148 | Tensor<[50, 768]> self = ?,<br>Tensor<[768, 3072]> mat2 = ?      | Done     | Done       | 0.9999683695135302 | 0      |
| 149 | Tensor<[50, 768]> self = ?,<br>Tensor<[768, 768]> mat2 = ?       | Done     | Done       | 0.9999688184068818 | 0      |
| 150 | Tensor<[512, 14]> self = ?,<br>Tensor<[14, 2048]> mat2 = ?       | Done     | Done       | 0.9999914815971065 | 0      |
| 151 | Tensor<[512, 14]> self = ?,<br>Tensor<[14, 512]> mat2 = ?        | Done     | Done       | 0.9999915040161153 | 0      |
| 152 | Tensor<[512, 1]> self = ?,<br>Tensor<[1, 768]> mat2 = ?          | Done     | Done       | 0.9999919512210418 | 0      |
| 153 | Tensor<[512, 256]> self = ?,<br>Tensor<[256, 256]> mat2 = ?      | Done     | Done       | 0.9999792499971015 | 0      |
| 154 | Tensor<[512, 256]> self = ?,<br>Tensor<[256, 768]> mat2 = ?      | Done     | Done       | 0.9999753790147148 | 0      |
| 155 | Tensor<[512, 2]> self = ?,<br>Tensor<[2, 512]> mat2 = ?          | Done     | Done       | 0.9999914717524295 | 0      |
| 156 | Tensor<[59, 1024]> self = ?,<br>Tensor<[1024, 512]> mat2 = ?     | Unknown  | Done       | 0.9999557542870553 | 0      |
| 157 | Tensor<[59, 512]> self = ?,<br>Tensor<[512, 1024]> mat2 = ?      | Unknown  | Done       | 0.9999713611579112 | 0      |
| 158 | Tensor<[59, 512]> self = ?,<br>Tensor<[512, 50272]> mat2 = ?     | Unknown  | Done       | 0.9999714613532341 | 0      |
| 159 | Tensor<[64, 1536]> self = ?,<br>Tensor<[1536, 768]> mat2 = ?     | Done     | Done       | 0.999960513429553  | 0      |
| 160 | Tensor<[64, 1]> self = ?,<br>Tensor<[1, 128]> mat2 = ?           | Done     | Done       | 0.9999912837175311 | 0      |
| 161 | Tensor<[64, 1]> self = ?,<br>Tensor<[1, 12]> mat2 = ?            | Done     | Done       | 0.9999930813902471 | 0      |
| 162 | Tensor<[64, 2048]> self = ?,<br>Tensor<[2048, 1024]> mat2 = ?    | Done     | Done       | 0.9999547015585127 | 0      |
| 163 | Tensor<[64, 256]> self = ?,<br>Tensor<[256, 4096]> mat2 = ?      | Done     | Done       | 0.999975408566765  | 0      |
| 164 | Tensor<[64, 256]> self = ?,<br>Tensor<[256, 64]> mat2 = ?        | Done     | Done       | 0.9999805154300316 | 0      |
| 165 | Tensor<[64, 4096]> self = ?,<br>Tensor<[4096, 256]> mat2 = ?     | Done     | Done       | 0.9998435096512214 | 0      |
| 166 | Tensor<[64, 4096]> self = ?,<br>Tensor<[4096, 64]> mat2 = ?      | Done     | Done       | 0.9998472241099299 | 0      |
| 167 | Tensor<[640, 1024]> self = ?,<br>Tensor<[1024, 160]> mat2 = ?    | Done     | Done       | 0.9999656728342653 | 0      |
| 168 | Tensor<[7, 18176]> self = ?,<br>Tensor<[18176, 4544]> mat2 = ?   | Unknown  | Unknown    | N/A                | N/A    |
| 169 | Tensor<[7, 4544]> self = ?,<br>Tensor<[4544, 18176]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |
| 170 | Tensor<[7, 4544]> self = ?,<br>Tensor<[4544, 4544]> mat2 = ?     | Unknown  | Unknown    | N/A                | N/A    |
| 171 | Tensor<[7, 4544]> self = ?,<br>Tensor<[4544, 4672]> mat2 = ?     | Unknown  | Unknown    | N/A                | N/A    |
| 172 | Tensor<[7, 4544]> self = ?,<br>Tensor<[4544, 65024]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |
| 173 | Tensor<[7, 768]> self = ?,<br>Tensor<[768, 2]> mat2 = ?          | Done     | Done       | 0.9999832819242689 | 0      |
| 174 | Tensor<[768, 196]> self = ?,<br>Tensor<[196, 3072]> mat2 = ?     | Done     | Done       | 0.9999821008199766 | 0      |
| 175 | Tensor<[768, 196]> self = ?,<br>Tensor<[196, 384]> mat2 = ?      | Done     | Done       | 0.9999820765339807 | 0      |
| 176 | Tensor<[768, 197]> self = ?,<br>Tensor<[197, 3072]> mat2 = ?     | Done     | Done       | 0.9999820525970593 | 0      |
| 177 | Tensor<[768, 197]> self = ?,<br>Tensor<[197, 768]> mat2 = ?      | Done     | Done       | 0.9999819578225038 | 0      |
| 178 | Tensor<[768, 50]> self = ?,<br>Tensor<[50, 3072]> mat2 = ?       | Done     | Done       | 0.9999859966622153 | 0      |
| 179 | Tensor<[768, 50]> self = ?,<br>Tensor<[50, 768]> mat2 = ?        | Done     | Done       | 0.9999859430390032 | 0      |
| 180 | Tensor<[784, 1]> self = ?,<br>Tensor<[1, 128]> mat2 = ?          | Done     | Done       | 0.9999925513526486 | 0      |
| 181 | Tensor<[784, 384]> self = ?,<br>Tensor<[384, 192]> mat2 = ?      | Done     | Done       | 0.9999731951460064 | 0      |
| 182 | Tensor<[784, 512]> self = ?,<br>Tensor<[512, 256]> mat2 = ?      | Done     | Done       | 0.9999714373659295 | 0      |
| 183 | Tensor<[9, 768]> self = ?,<br>Tensor<[768, 1280]> mat2 = ?       | Unknown  | Done       | 0.9999697353726658 | 0      |
| 184 | Tensor<[9, 768]> self = ?,<br>Tensor<[768, 320]> mat2 = ?        | Unknown  | Done       | 0.9999685241790405 | 0      |
| 185 | Tensor<[9, 768]> self = ?,<br>Tensor<[768, 640]> mat2 = ?        | Unknown  | Done       | 0.9999692919937904 | 0      |
| 186 | Tensor<[920, 256]> self = ?,<br>Tensor<[256, 256]> mat2 = ?      | Done     | Done       | 0.9999752895003104 | 0      |
| 187 | Tensor<[s0*s1, 1280]> self = ?,<br>Tensor<[1280, 1280]> mat2 = ? | Unknown  | Unknown    | N/A                | N/A    |
| 188 | Tensor<[s0*s1, 640]> self = ?,<br>Tensor<[640, 640]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |
| 189 | Tensor<[s1*s2, 1280]> self = ?,<br>Tensor<[1280, 1280]> mat2 = ? | Unknown  | Unknown    | N/A                | N/A    |
| 190 | Tensor<[s1*s2, 320]> self = ?,<br>Tensor<[320, 320]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |
| 191 | Tensor<[s1*s2, 640]> self = ?,<br>Tensor<[640, 640]> mat2 = ?    | Unknown  | Unknown    | N/A                | N/A    |

