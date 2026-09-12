# synth_Town05_O_day step4 three-layer evaluation (test split)

## seed0 ipm->geo (C1 forward geometry)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.0451 m  CI95 [-1.4790,-0.8569] SIG  (ipm 1.536 -> geo 0.491)
 C01 dp90    = -1.3214 m  CI95 [-1.5650,-1.0447] SIG  (ipm 2.006 -> geo 0.684)
 C01 drmse   = -1.1558 m  CI95 [-1.4118,-0.9768] SIG  (ipm 1.706 -> geo 0.550)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.4079 m  CI95 [-1.4830,-0.9350] SIG  (ipm 1.741 -> geo 0.333)
 C02 dp90    = -1.0181 m  CI95 [-1.6082,-0.1435] SIG  (ipm 2.189 -> geo 1.170)
 C02 drmse   = -0.9754 m  CI95 [-1.3288,-0.5036] SIG  (ipm 1.716 -> geo 0.741)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.1842 m  CI95 [-1.4534,-0.9347] SIG  (ipm 1.689 -> geo 0.505)
 C03 dp90    = -1.1213 m  CI95 [-2.0204,-0.4230] SIG  (ipm 2.612 -> geo 1.490)
 C03 drmse   = -1.0497 m  CI95 [-1.4477,-0.6335] SIG  (ipm 1.863 -> geo 0.813)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5926 m  CI95 [-2.0489,-1.1230] SIG  (ipm 1.952 -> geo 0.359)
 C04 dp90    = -2.0465 m  CI95 [-2.0846,-1.2828] SIG  (ipm 3.158 -> geo 1.112)
 C04 drmse   = -1.5257 m  CI95 [-1.7379,-1.1715] SIG  (ipm 2.236 -> geo 0.710)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.4925 m  CI95 [-1.6611,-0.5258] SIG  (ipm 2.376 -> geo 0.884)
 C05 dp90    = -1.4073 m  CI95 [-1.5247,-0.8668] SIG  (ipm 2.555 -> geo 1.148)
 C05 drmse   = -1.3187 m  CI95 [-1.4933,-0.7447] SIG  (ipm 2.251 -> geo 0.932)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.9704 m  CI95 [-1.1272,+0.5946] ns  (ipm 1.716 -> geo 0.746)
 C06 dp90    = -0.3393 m  CI95 [-1.2135,-0.0228] SIG  (ipm 1.945 -> geo 1.605)
 C06 drmse   = -0.4550 m  CI95 [-1.1162,+0.2944] ns  (ipm 1.574 -> geo 1.119)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.1756 m  CI95 [-1.3691,-0.9747] SIG  (ipm 1.795 -> geo 0.619)
 ALL dp90    = -1.0943 m  CI95 [-1.7561,-0.7041] SIG  (ipm 2.539 -> geo 1.444)
 ALL drmse   = -1.0825 m  CI95 [-1.3024,-0.8171] SIG  (ipm 1.916 -> geo 0.833)

## seed0 geo->final (C2 residual learning)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -0.0074 m  CI95 [-0.0864,+0.0436] ns  (geo 0.491 -> final 0.484)
 C01 dp90    = +0.0410 m  CI95 [-0.0704,+0.0441] ns  (geo 0.684 -> final 0.725)
 C01 drmse   = -0.0045 m  CI95 [-0.0543,+0.0242] ns  (geo 0.550 -> final 0.546)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = +0.0439 m  CI95 [-0.0194,+0.0764] ns  (geo 0.333 -> final 0.377)
 C02 dp90    = +0.0252 m  CI95 [-0.0691,+0.2614] ns  (geo 1.170 -> final 1.196)
 C02 drmse   = +0.0900 m  CI95 [+0.0249,+0.1987] SIG  (geo 0.741 -> final 0.831)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -0.0051 m  CI95 [-0.0502,+0.0457] ns  (geo 0.505 -> final 0.500)
 C03 dp90    = -0.0463 m  CI95 [-0.2283,+0.0932] ns  (geo 1.490 -> final 1.444)
 C03 drmse   = -0.0132 m  CI95 [-0.0665,+0.0471] ns  (geo 0.813 -> final 0.800)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = +0.0620 m  CI95 [-0.0114,+0.1703] ns  (geo 0.359 -> final 0.421)
 C04 dp90    = +0.1588 m  CI95 [-0.0414,+0.2009] ns  (geo 1.112 -> final 1.271)
 C04 drmse   = +0.0803 m  CI95 [+0.0075,+0.1456] SIG  (geo 0.710 -> final 0.790)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -0.1666 m  CI95 [-0.2838,+0.0268] ns  (geo 0.884 -> final 0.717)
 C05 dp90    = -0.1083 m  CI95 [-0.3470,-0.0049] SIG  (geo 1.148 -> final 1.039)
 C05 drmse   = -0.1785 m  CI95 [-0.2202,-0.0635] SIG  (geo 0.932 -> final 0.754)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.1617 m  CI95 [-0.1903,+0.0915] ns  (geo 0.746 -> final 0.584)
 C06 dp90    = -0.1010 m  CI95 [-0.1370,+0.1072] ns  (geo 1.605 -> final 1.504)
 C06 drmse   = -0.0783 m  CI95 [-0.1437,+0.0503] ns  (geo 1.119 -> final 1.041)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -0.0727 m  CI95 [-0.1585,+0.0314] ns  (geo 0.619 -> final 0.547)
 ALL dp90    = -0.0603 m  CI95 [-0.1448,+0.1299] ns  (geo 1.444 -> final 1.384)
 ALL drmse   = -0.0295 m  CI95 [-0.0722,+0.0192] ns  (geo 0.833 -> final 0.804)

## seed0 ipm->final (total)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.0525 m  CI95 [-1.5638,-0.8135] SIG  (ipm 1.536 -> final 0.484)
 C01 dp90    = -1.2804 m  CI95 [-1.6157,-0.9786] SIG  (ipm 2.006 -> final 0.725)
 C01 drmse   = -1.1603 m  CI95 [-1.4644,-0.9513] SIG  (ipm 1.706 -> final 0.546)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.3640 m  CI95 [-1.4449,-0.8445] SIG  (ipm 1.741 -> final 0.377)
 C02 dp90    = -0.9929 m  CI95 [-1.5330,+0.0278] ns  (ipm 2.189 -> final 1.196)
 C02 drmse   = -0.8853 m  CI95 [-1.2545,-0.3440] SIG  (ipm 1.716 -> final 0.831)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.1893 m  CI95 [-1.4271,-0.9553] SIG  (ipm 1.689 -> final 0.500)
 C03 dp90    = -1.1676 m  CI95 [-1.9237,-0.4476] SIG  (ipm 2.612 -> final 1.444)
 C03 drmse   = -1.0628 m  CI95 [-1.4140,-0.6829] SIG  (ipm 1.863 -> final 0.800)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5307 m  CI95 [-1.8780,-1.1042] SIG  (ipm 1.952 -> final 0.421)
 C04 dp90    = -1.8877 m  CI95 [-1.8975,-1.2795] SIG  (ipm 3.158 -> final 1.271)
 C04 drmse   = -1.4454 m  CI95 [-1.5901,-1.1492] SIG  (ipm 2.236 -> final 0.790)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.6591 m  CI95 [-1.9397,-0.5532] SIG  (ipm 2.376 -> final 0.717)
 C05 dp90    = -1.5157 m  CI95 [-1.6690,-1.1531] SIG  (ipm 2.555 -> final 1.039)
 C05 drmse   = -1.4973 m  CI95 [-1.6936,-0.8549] SIG  (ipm 2.251 -> final 0.754)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -1.1321 m  CI95 [-1.3144,+0.4771] ns  (ipm 1.716 -> final 0.584)
 C06 dp90    = -0.4403 m  CI95 [-1.1933,-0.0702] SIG  (ipm 1.945 -> final 1.504)
 C06 drmse   = -0.5333 m  CI95 [-1.2133,+0.2436] ns  (ipm 1.574 -> final 1.041)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.2482 m  CI95 [-1.4283,-1.0102] SIG  (ipm 1.795 -> final 0.547)
 ALL dp90    = -1.1547 m  CI95 [-1.7112,-0.8170] SIG  (ipm 2.539 -> final 1.384)
 ALL drmse   = -1.1121 m  CI95 [-1.3316,-0.8459] SIG  (ipm 1.916 -> final 0.804)

## seed1 ipm->geo (C1 forward geometry)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.0451 m  CI95 [-1.4801,-0.8572] SIG  (ipm 1.536 -> geo 0.491)
 C01 dp90    = -1.3214 m  CI95 [-1.5444,-1.0657] SIG  (ipm 2.006 -> geo 0.684)
 C01 drmse   = -1.1558 m  CI95 [-1.4079,-0.9826] SIG  (ipm 1.706 -> geo 0.550)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.4079 m  CI95 [-1.4864,-0.9229] SIG  (ipm 1.741 -> geo 0.333)
 C02 dp90    = -1.0181 m  CI95 [-1.6285,-0.1507] SIG  (ipm 2.189 -> geo 1.170)
 C02 drmse   = -0.9754 m  CI95 [-1.3285,-0.4872] SIG  (ipm 1.716 -> geo 0.741)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.1842 m  CI95 [-1.4705,-0.9307] SIG  (ipm 1.689 -> geo 0.505)
 C03 dp90    = -1.1213 m  CI95 [-1.9836,-0.4343] SIG  (ipm 2.612 -> geo 1.490)
 C03 drmse   = -1.0497 m  CI95 [-1.4640,-0.6611] SIG  (ipm 1.863 -> geo 0.813)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5926 m  CI95 [-2.0561,-1.1346] SIG  (ipm 1.952 -> geo 0.359)
 C04 dp90    = -2.0465 m  CI95 [-2.0836,-1.2536] SIG  (ipm 3.158 -> geo 1.112)
 C04 drmse   = -1.5257 m  CI95 [-1.7472,-1.1573] SIG  (ipm 2.236 -> geo 0.710)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.4925 m  CI95 [-1.6608,-0.4991] SIG  (ipm 2.376 -> geo 0.884)
 C05 dp90    = -1.4073 m  CI95 [-1.5144,-0.8744] SIG  (ipm 2.555 -> geo 1.148)
 C05 drmse   = -1.3187 m  CI95 [-1.4854,-0.7339] SIG  (ipm 2.251 -> geo 0.932)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.9704 m  CI95 [-1.1367,+0.5973] ns  (ipm 1.716 -> geo 0.746)
 C06 dp90    = -0.3393 m  CI95 [-1.2011,-0.0015] SIG  (ipm 1.945 -> geo 1.605)
 C06 drmse   = -0.4550 m  CI95 [-1.1080,+0.3064] ns  (ipm 1.574 -> geo 1.119)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.1756 m  CI95 [-1.3644,-0.9827] SIG  (ipm 1.795 -> geo 0.619)
 ALL dp90    = -1.0943 m  CI95 [-1.8021,-0.7193] SIG  (ipm 2.539 -> geo 1.444)
 ALL drmse   = -1.0825 m  CI95 [-1.3167,-0.8245] SIG  (ipm 1.916 -> geo 0.833)

## seed1 geo->final (C2 residual learning)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -0.0604 m  CI95 [-0.1005,+0.1406] ns  (geo 0.491 -> final 0.431)
 C01 dp90    = +0.1360 m  CI95 [-0.0523,+0.1400] ns  (geo 0.684 -> final 0.820)
 C01 drmse   = +0.0347 m  CI95 [-0.0544,+0.0980] ns  (geo 0.550 -> final 0.585)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = +0.0196 m  CI95 [-0.0171,+0.0413] ns  (geo 0.333 -> final 0.353)
 C02 dp90    = +0.0911 m  CI95 [-0.3044,+0.1520] ns  (geo 1.170 -> final 1.262)
 C02 drmse   = +0.0465 m  CI95 [-0.0558,+0.1717] ns  (geo 0.741 -> final 0.788)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -0.0167 m  CI95 [-0.0903,+0.0167] ns  (geo 0.505 -> final 0.488)
 C03 dp90    = -0.1138 m  CI95 [-0.1925,-0.0052] SIG  (geo 1.490 -> final 1.376)
 C03 drmse   = -0.0518 m  CI95 [-0.0717,-0.0223] SIG  (geo 0.813 -> final 0.762)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = +0.0542 m  CI95 [+0.0003,+0.0889] SIG  (geo 0.359 -> final 0.413)
 C04 dp90    = +0.0666 m  CI95 [-0.0240,+0.0824] ns  (geo 1.112 -> final 1.179)
 C04 drmse   = +0.0446 m  CI95 [+0.0193,+0.0613] SIG  (geo 0.710 -> final 0.755)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -0.1350 m  CI95 [-0.2388,-0.0010] SIG  (geo 0.884 -> final 0.749)
 C05 dp90    = -0.1186 m  CI95 [-0.3599,-0.0524] SIG  (geo 1.148 -> final 1.029)
 C05 drmse   = -0.1652 m  CI95 [-0.1954,-0.0817] SIG  (geo 0.932 -> final 0.767)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.1092 m  CI95 [-0.5043,+0.0162] ns  (geo 0.746 -> final 0.637)
 C06 dp90    = -0.4854 m  CI95 [-0.5015,+0.1203] ns  (geo 1.605 -> final 1.120)
 C06 drmse   = -0.2312 m  CI95 [-0.3696,+0.0285] ns  (geo 1.119 -> final 0.888)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -0.0743 m  CI95 [-0.1272,+0.0251] ns  (geo 0.619 -> final 0.545)
 ALL dp90    = -0.3031 m  CI95 [-0.4684,+0.0308] ns  (geo 1.444 -> final 1.141)
 ALL drmse   = -0.0732 m  CI95 [-0.1377,-0.0104] SIG  (geo 0.833 -> final 0.760)

## seed1 ipm->final (total)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.1055 m  CI95 [-1.5762,-0.7169] SIG  (ipm 1.536 -> final 0.431)
 C01 dp90    = -1.1854 m  CI95 [-1.5618,-0.9332] SIG  (ipm 2.006 -> final 0.820)
 C01 drmse   = -1.1211 m  CI95 [-1.4425,-0.8824] SIG  (ipm 1.706 -> final 0.585)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.3883 m  CI95 [-1.4769,-0.8401] SIG  (ipm 1.741 -> final 0.353)
 C02 dp90    = -0.9270 m  CI95 [-1.5515,-0.2105] SIG  (ipm 2.189 -> final 1.262)
 C02 drmse   = -0.9289 m  CI95 [-1.2919,-0.3546] SIG  (ipm 1.716 -> final 0.788)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.2009 m  CI95 [-1.5212,-1.0241] SIG  (ipm 1.689 -> final 0.488)
 C03 dp90    = -1.2352 m  CI95 [-2.0330,-0.5349] SIG  (ipm 2.612 -> final 1.376)
 C03 drmse   = -1.1014 m  CI95 [-1.4933,-0.6998] SIG  (ipm 1.863 -> final 0.762)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5385 m  CI95 [-1.9444,-1.0625] SIG  (ipm 1.952 -> final 0.413)
 C04 dp90    = -1.9799 m  CI95 [-2.0103,-1.2425] SIG  (ipm 3.158 -> final 1.179)
 C04 drmse   = -1.4810 m  CI95 [-1.6849,-1.1127] SIG  (ipm 2.236 -> final 0.755)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.6275 m  CI95 [-1.8935,-0.5674] SIG  (ipm 2.376 -> final 0.749)
 C05 dp90    = -1.5260 m  CI95 [-1.6427,-1.2011] SIG  (ipm 2.555 -> final 1.029)
 C05 drmse   = -1.4839 m  CI95 [-1.6796,-0.8596] SIG  (ipm 2.251 -> final 0.767)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -1.0795 m  CI95 [-1.2361,+0.0976] ns  (ipm 1.716 -> final 0.637)
 C06 dp90    = -0.8247 m  CI95 [-1.2280,-0.3157] SIG  (ipm 1.945 -> final 1.120)
 C06 drmse   = -0.6862 m  CI95 [-1.1717,-0.0506] SIG  (ipm 1.574 -> final 0.888)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.2499 m  CI95 [-1.4426,-0.9845] SIG  (ipm 1.795 -> final 0.545)
 ALL dp90    = -1.3975 m  CI95 [-1.8110,-1.0240] SIG  (ipm 2.539 -> final 1.141)
 ALL drmse   = -1.1557 m  CI95 [-1.3602,-0.9334] SIG  (ipm 1.916 -> final 0.760)

## seed2 ipm->geo (C1 forward geometry)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.0451 m  CI95 [-1.4783,-0.8570] SIG  (ipm 1.536 -> geo 0.491)
 C01 dp90    = -1.3214 m  CI95 [-1.5415,-1.0478] SIG  (ipm 2.006 -> geo 0.684)
 C01 drmse   = -1.1558 m  CI95 [-1.4068,-0.9781] SIG  (ipm 1.706 -> geo 0.550)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.4079 m  CI95 [-1.4832,-0.8465] SIG  (ipm 1.741 -> geo 0.333)
 C02 dp90    = -1.0181 m  CI95 [-1.6132,-0.1290] SIG  (ipm 2.189 -> geo 1.170)
 C02 drmse   = -0.9754 m  CI95 [-1.3135,-0.4797] SIG  (ipm 1.716 -> geo 0.741)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.1842 m  CI95 [-1.4516,-0.9873] SIG  (ipm 1.689 -> geo 0.505)
 C03 dp90    = -1.1213 m  CI95 [-2.0084,-0.4445] SIG  (ipm 2.612 -> geo 1.490)
 C03 drmse   = -1.0497 m  CI95 [-1.4500,-0.6797] SIG  (ipm 1.863 -> geo 0.813)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5926 m  CI95 [-2.0264,-1.1381] SIG  (ipm 1.952 -> geo 0.359)
 C04 dp90    = -2.0465 m  CI95 [-2.0837,-1.2695] SIG  (ipm 3.158 -> geo 1.112)
 C04 drmse   = -1.5257 m  CI95 [-1.7351,-1.1666] SIG  (ipm 2.236 -> geo 0.710)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.4925 m  CI95 [-1.6592,-0.4852] SIG  (ipm 2.376 -> geo 0.884)
 C05 dp90    = -1.4073 m  CI95 [-1.5310,-0.8362] SIG  (ipm 2.555 -> geo 1.148)
 C05 drmse   = -1.3187 m  CI95 [-1.4846,-0.7202] SIG  (ipm 2.251 -> geo 0.932)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.9704 m  CI95 [-1.1871,+0.5982] ns  (ipm 1.716 -> geo 0.746)
 C06 dp90    = -0.3393 m  CI95 [-1.2085,-0.0458] SIG  (ipm 1.945 -> geo 1.605)
 C06 drmse   = -0.4550 m  CI95 [-1.1144,+0.2974] ns  (ipm 1.574 -> geo 1.119)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.1756 m  CI95 [-1.3691,-0.9817] SIG  (ipm 1.795 -> geo 0.619)
 ALL dp90    = -1.0943 m  CI95 [-1.7846,-0.7448] SIG  (ipm 2.539 -> geo 1.444)
 ALL drmse   = -1.0825 m  CI95 [-1.3112,-0.8199] SIG  (ipm 1.916 -> geo 0.833)

## seed2 geo->final (C2 residual learning)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -0.0039 m  CI95 [-0.0663,+0.0616] ns  (geo 0.491 -> final 0.487)
 C01 dp90    = +0.0389 m  CI95 [+0.0139,+0.1002] SIG  (geo 0.684 -> final 0.723)
 C01 drmse   = +0.0185 m  CI95 [-0.0099,+0.0406] ns  (geo 0.550 -> final 0.569)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = +0.0657 m  CI95 [-0.0001,+0.1012] ns  (geo 0.333 -> final 0.399)
 C02 dp90    = +0.1294 m  CI95 [+0.0310,+0.4196] SIG  (geo 1.170 -> final 1.300)
 C02 drmse   = +0.1124 m  CI95 [+0.0542,+0.2020] SIG  (geo 0.741 -> final 0.853)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -0.0495 m  CI95 [-0.0604,+0.0714] ns  (geo 0.505 -> final 0.455)
 C03 dp90    = -0.1762 m  CI95 [-0.3607,+0.3238] ns  (geo 1.490 -> final 1.314)
 C03 drmse   = -0.0485 m  CI95 [-0.1406,+0.0908] ns  (geo 0.813 -> final 0.765)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = +0.0229 m  CI95 [-0.0510,+0.0467] ns  (geo 0.359 -> final 0.382)
 C04 dp90    = +0.0413 m  CI95 [-0.0741,+0.0587] ns  (geo 1.112 -> final 1.153)
 C04 drmse   = +0.0149 m  CI95 [-0.0169,+0.0334] ns  (geo 0.710 -> final 0.725)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -0.1350 m  CI95 [-0.2222,-0.0014] SIG  (geo 0.884 -> final 0.749)
 C05 dp90    = -0.0790 m  CI95 [-0.2882,+0.0639] ns  (geo 1.148 -> final 1.069)
 C05 drmse   = -0.1337 m  CI95 [-0.1711,-0.0522] SIG  (geo 0.932 -> final 0.798)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -0.1054 m  CI95 [-0.2094,+0.0871] ns  (geo 0.746 -> final 0.640)
 C06 dp90    = -0.1811 m  CI95 [-0.1836,+0.1133] ns  (geo 1.605 -> final 1.424)
 C06 drmse   = -0.1023 m  CI95 [-0.1431,+0.0614] ns  (geo 1.119 -> final 1.017)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -0.0716 m  CI95 [-0.1280,+0.0241] ns  (geo 0.619 -> final 0.548)
 ALL dp90    = -0.1252 m  CI95 [-0.2075,+0.0764] ns  (geo 1.444 -> final 1.319)
 ALL drmse   = -0.0447 m  CI95 [-0.0821,+0.0056] ns  (geo 0.833 -> final 0.789)

## seed2 ipm->final (total)
 C01 n_obs=1269 n_tracks=17
 C01 dmedian = -1.0490 m  CI95 [-1.5439,-0.8200] SIG  (ipm 1.536 -> final 0.487)
 C01 dp90    = -1.2825 m  CI95 [-1.4883,-0.9932] SIG  (ipm 2.006 -> final 0.723)
 C01 drmse   = -1.1373 m  CI95 [-1.4020,-0.9517] SIG  (ipm 1.706 -> final 0.569)
 C02 n_obs=593 n_tracks=19
 C02 dmedian = -1.3422 m  CI95 [-1.4401,-0.7987] SIG  (ipm 1.741 -> final 0.399)
 C02 dp90    = -0.8887 m  CI95 [-1.5210,+0.1721] ns  (ipm 2.189 -> final 1.300)
 C02 drmse   = -0.8630 m  CI95 [-1.2337,-0.3245] SIG  (ipm 1.716 -> final 0.853)
 C03 n_obs=2682 n_tracks=36
 C03 dmedian = -1.2337 m  CI95 [-1.4938,-0.9868] SIG  (ipm 1.689 -> final 0.455)
 C03 dp90    = -1.2976 m  CI95 [-1.7035,-0.6017] SIG  (ipm 2.612 -> final 1.314)
 C03 drmse   = -1.0982 m  CI95 [-1.3906,-0.7779] SIG  (ipm 1.863 -> final 0.765)
 C04 n_obs=1312 n_tracks=25
 C04 dmedian = -1.5697 m  CI95 [-2.0041,-1.1211] SIG  (ipm 1.952 -> final 0.382)
 C04 dp90    = -2.0052 m  CI95 [-2.0472,-1.3283] SIG  (ipm 3.158 -> final 1.153)
 C04 drmse   = -1.5107 m  CI95 [-1.7008,-1.1852] SIG  (ipm 2.236 -> final 0.725)
 C05 n_obs=1222 n_tracks=17
 C05 dmedian = -1.6275 m  CI95 [-1.8729,-0.5667] SIG  (ipm 2.376 -> final 0.749)
 C05 dp90    = -1.4863 m  CI95 [-1.5878,-1.1524] SIG  (ipm 2.555 -> final 1.069)
 C05 drmse   = -1.4524 m  CI95 [-1.6361,-0.8625] SIG  (ipm 2.251 -> final 0.798)
 C06 n_obs=1271 n_tracks=17
 C06 dmedian = -1.0757 m  CI95 [-1.2338,+0.3999] ns  (ipm 1.716 -> final 0.640)
 C06 dp90    = -0.5204 m  CI95 [-1.2243,-0.1584] SIG  (ipm 1.945 -> final 1.424)
 C06 drmse   = -0.5573 m  CI95 [-1.1662,+0.1624] ns  (ipm 1.574 -> final 1.017)
 ALL n_obs=8349 n_tracks=131
 ALL dmedian = -1.2472 m  CI95 [-1.4284,-0.9819] SIG  (ipm 1.795 -> final 0.548)
 ALL dp90    = -1.2195 m  CI95 [-1.7316,-0.9297] SIG  (ipm 2.539 -> final 1.319)
 ALL drmse   = -1.1272 m  CI95 [-1.3291,-0.8863] SIG  (ipm 1.916 -> final 0.789)

## identity metrics on test window (baseline geometric remerge vs seeds)
 gate     method     DetA     AssA     LocA     HOTA     IDF1     MOTA   IDsw
  0.5   baseline   0.1504   0.3936   0.4219   0.2433   0.2160  -0.2722     81
  0.5      seed0   0.1850   0.4764   0.3930   0.2968   0.2662  -0.1647     74
  0.5      seed1   0.1776   0.4352   0.3895   0.2780   0.2575  -0.1747     68
  0.5      seed2   0.1777   0.4510   0.3789   0.2831   0.2598  -0.1665     64
  1.0   baseline   0.2808   0.4413   0.5499   0.3520   0.3371   0.0252    130
  1.0      seed0   0.2811   0.4639   0.5771   0.3611   0.3433   0.0420    130
  1.0      seed1   0.2841   0.4628   0.5779   0.3626   0.3454   0.0544    124
  1.0      seed2   0.2762   0.4356   0.5687   0.3468   0.3326   0.0445    123
  1.5   baseline   0.3157   0.4387   0.6544   0.3722   0.3604   0.0933    154
  1.5      seed0   0.3522   0.4462   0.6271   0.3964   0.3903   0.1775    153
  1.5      seed1   0.3673   0.4719   0.6326   0.4164   0.4168   0.2111    141
  1.5      seed2   0.3744   0.4547   0.6000   0.4126   0.4164   0.2278    147
  2.0   baseline   0.3762   0.4538   0.6740   0.4132   0.4134   0.2062    166
  2.0      seed0   0.3841   0.4558   0.6867   0.4184   0.4180   0.2340    159
  2.0      seed1   0.3886   0.4777   0.7027   0.4309   0.4315   0.2477    148
  2.0      seed2   0.3914   0.4615   0.6836   0.4250   0.4268   0.2564    156

## BEV-HOTA (mean over 0.5/1/1.5/2 m gates)
  baseline  BEV-HOTA = 0.3452
     seed0  BEV-HOTA = 0.3682
     seed1  BEV-HOTA = 0.3720
     seed2  BEV-HOTA = 0.3669
