# lumpi_M3 step4 three-layer evaluation (test split)

## seed0 ipm->geo (C1 forward geometry)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9772 m  CI95 [-1.0954,-0.6498] SIG  (ipm 1.788 -> geo 0.811)
 C08 dp90    = -0.8003 m  CI95 [-1.0151,-0.7202] SIG  (ipm 2.861 -> geo 2.060)
 C08 drmse   = -0.8084 m  CI95 [-0.9160,-0.6690] SIG  (ipm 1.986 -> geo 1.178)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1556 m  CI95 [-1.3944,-1.0185] SIG  (ipm 2.008 -> geo 0.853)
 C09 dp90    = -1.4379 m  CI95 [-1.5467,-0.6828] SIG  (ipm 3.226 -> geo 1.788)
 C09 drmse   = -1.1464 m  CI95 [-1.3095,-0.9045] SIG  (ipm 2.278 -> geo 1.132)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.1778 m  CI95 [-1.3521,-0.9544] SIG  (ipm 1.909 -> geo 0.732)
 C10 dp90    = -1.2910 m  CI95 [-1.4420,-0.6052] SIG  (ipm 2.975 -> geo 1.684)
 C10 drmse   = -1.0493 m  CI95 [-1.1870,-0.8818] SIG  (ipm 2.133 -> geo 1.084)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.1373 m  CI95 [-1.2236,-1.0176] SIG  (ipm 1.931 -> geo 0.793)
 ALL dp90    = -1.2217 m  CI95 [-1.4561,-0.8422] SIG  (ipm 3.002 -> geo 1.781)
 ALL drmse   = -1.0427 m  CI95 [-1.1501,-0.9301] SIG  (ipm 2.165 -> geo 1.122)

## seed0 geo->final (C2 residual learning)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.0070 m  CI95 [-0.2029,+0.0610] ns  (geo 0.811 -> final 0.804)
 C08 dp90    = +0.0857 m  CI95 [-0.0444,+0.0894] ns  (geo 2.060 -> final 2.146)
 C08 drmse   = +0.0028 m  CI95 [-0.0508,+0.0477] ns  (geo 1.178 -> final 1.181)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = +0.0361 m  CI95 [-0.0658,+0.0966] ns  (geo 0.853 -> final 0.889)
 C09 dp90    = -0.0728 m  CI95 [-0.1127,+0.0391] ns  (geo 1.788 -> final 1.715)
 C09 drmse   = -0.0354 m  CI95 [-0.0637,+0.0073] ns  (geo 1.132 -> final 1.096)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = +0.0934 m  CI95 [-0.0019,+0.1633] ns  (geo 0.732 -> final 0.825)
 C10 dp90    = -0.0442 m  CI95 [-0.0538,+0.1544] ns  (geo 1.684 -> final 1.640)
 C10 drmse   = +0.0332 m  CI95 [-0.0051,+0.0816] ns  (geo 1.084 -> final 1.117)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = +0.0504 m  CI95 [-0.0404,+0.0880] ns  (geo 0.793 -> final 0.844)
 ALL dp90    = -0.0647 m  CI95 [-0.0755,+0.0480] ns  (geo 1.781 -> final 1.716)
 ALL drmse   = -0.0003 m  CI95 [-0.0247,+0.0265] ns  (geo 1.122 -> final 1.122)

## seed0 ipm->final (total)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9841 m  CI95 [-1.2704,-0.6095] SIG  (ipm 1.788 -> final 0.804)
 C08 dp90    = -0.7146 m  CI95 [-1.0384,-0.6266] SIG  (ipm 2.861 -> final 2.146)
 C08 drmse   = -0.8056 m  CI95 [-0.9426,-0.6382] SIG  (ipm 1.986 -> final 1.181)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1196 m  CI95 [-1.3968,-0.9862] SIG  (ipm 2.008 -> final 0.889)
 C09 dp90    = -1.5108 m  CI95 [-1.5744,-0.7583] SIG  (ipm 3.226 -> final 1.715)
 C09 drmse   = -1.1818 m  CI95 [-1.3377,-0.9474] SIG  (ipm 2.278 -> final 1.096)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.0844 m  CI95 [-1.3155,-0.8559] SIG  (ipm 1.909 -> final 0.825)
 C10 dp90    = -1.3352 m  CI95 [-1.3877,-0.4732] SIG  (ipm 2.975 -> final 1.640)
 C10 drmse   = -1.0161 m  CI95 [-1.1766,-0.8049] SIG  (ipm 2.133 -> final 1.117)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.0869 m  CI95 [-1.2265,-0.9551] SIG  (ipm 1.931 -> final 0.844)
 ALL dp90    = -1.2864 m  CI95 [-1.5225,-0.8059] SIG  (ipm 3.002 -> final 1.716)
 ALL drmse   = -1.0429 m  CI95 [-1.1558,-0.9036] SIG  (ipm 2.165 -> final 1.122)

## seed1 ipm->geo (C1 forward geometry)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9772 m  CI95 [-1.0941,-0.6557] SIG  (ipm 1.788 -> geo 0.811)
 C08 dp90    = -0.8003 m  CI95 [-1.0063,-0.7015] SIG  (ipm 2.861 -> geo 2.060)
 C08 drmse   = -0.8084 m  CI95 [-0.9042,-0.6745] SIG  (ipm 1.986 -> geo 1.178)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1556 m  CI95 [-1.3969,-0.9820] SIG  (ipm 2.008 -> geo 0.853)
 C09 dp90    = -1.4379 m  CI95 [-1.5431,-0.6771] SIG  (ipm 3.226 -> geo 1.788)
 C09 drmse   = -1.1464 m  CI95 [-1.3079,-0.9058] SIG  (ipm 2.278 -> geo 1.132)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.1778 m  CI95 [-1.3732,-0.9448] SIG  (ipm 1.909 -> geo 0.732)
 C10 dp90    = -1.2910 m  CI95 [-1.4545,-0.5989] SIG  (ipm 2.975 -> geo 1.684)
 C10 drmse   = -1.0493 m  CI95 [-1.1934,-0.8660] SIG  (ipm 2.133 -> geo 1.084)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.1373 m  CI95 [-1.2237,-1.0187] SIG  (ipm 1.931 -> geo 0.793)
 ALL dp90    = -1.2217 m  CI95 [-1.4601,-0.8469] SIG  (ipm 3.002 -> geo 1.781)
 ALL drmse   = -1.0427 m  CI95 [-1.1549,-0.9244] SIG  (ipm 2.165 -> geo 1.122)

## seed1 geo->final (C2 residual learning)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.0141 m  CI95 [-0.1844,+0.0789] ns  (geo 0.811 -> final 0.797)
 C08 dp90    = +0.1897 m  CI95 [-0.0010,+0.2070] ns  (geo 2.060 -> final 2.250)
 C08 drmse   = +0.0389 m  CI95 [-0.0223,+0.0953] ns  (geo 1.178 -> final 1.217)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -0.0290 m  CI95 [-0.0933,+0.0714] ns  (geo 0.853 -> final 0.824)
 C09 dp90    = -0.0657 m  CI95 [-0.1737,+0.0476] ns  (geo 1.788 -> final 1.722)
 C09 drmse   = -0.0404 m  CI95 [-0.0770,+0.0091] ns  (geo 1.132 -> final 1.091)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = +0.0838 m  CI95 [+0.0037,+0.1458] SIG  (geo 0.732 -> final 0.815)
 C10 dp90    = -0.0512 m  CI95 [-0.0774,+0.1217] ns  (geo 1.684 -> final 1.632)
 C10 drmse   = +0.0181 m  CI95 [-0.0202,+0.0689] ns  (geo 1.084 -> final 1.102)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = +0.0213 m  CI95 [-0.0481,+0.0793] ns  (geo 0.793 -> final 0.815)
 ALL dp90    = -0.0569 m  CI95 [-0.1246,+0.0425] ns  (geo 1.781 -> final 1.724)
 ALL drmse   = -0.0007 m  CI95 [-0.0265,+0.0311] ns  (geo 1.122 -> final 1.121)

## seed1 ipm->final (total)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9913 m  CI95 [-1.2584,-0.6271] SIG  (ipm 1.788 -> final 0.797)
 C08 dp90    = -0.6107 m  CI95 [-0.9643,-0.5895] SIG  (ipm 2.861 -> final 2.250)
 C08 drmse   = -0.7695 m  CI95 [-0.9176,-0.6234] SIG  (ipm 1.986 -> final 1.217)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1846 m  CI95 [-1.3899,-0.9922] SIG  (ipm 2.008 -> final 0.824)
 C09 dp90    = -1.5036 m  CI95 [-1.5647,-0.8420] SIG  (ipm 3.226 -> final 1.722)
 C09 drmse   = -1.1868 m  CI95 [-1.3351,-0.9731] SIG  (ipm 2.278 -> final 1.091)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.0940 m  CI95 [-1.3242,-0.8657] SIG  (ipm 1.909 -> final 0.815)
 C10 dp90    = -1.3422 m  CI95 [-1.4053,-0.5331] SIG  (ipm 2.975 -> final 1.632)
 C10 drmse   = -1.0313 m  CI95 [-1.1807,-0.8197] SIG  (ipm 2.133 -> final 1.102)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.1160 m  CI95 [-1.2180,-0.9633] SIG  (ipm 1.931 -> final 0.815)
 ALL dp90    = -1.2786 m  CI95 [-1.5198,-0.8992] SIG  (ipm 3.002 -> final 1.724)
 ALL drmse   = -1.0433 m  CI95 [-1.1542,-0.9219] SIG  (ipm 2.165 -> final 1.121)

## seed2 ipm->geo (C1 forward geometry)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9772 m  CI95 [-1.0981,-0.6694] SIG  (ipm 1.788 -> geo 0.811)
 C08 dp90    = -0.8003 m  CI95 [-1.0000,-0.6555] SIG  (ipm 2.861 -> geo 2.060)
 C08 drmse   = -0.8084 m  CI95 [-0.9103,-0.6751] SIG  (ipm 1.986 -> geo 1.178)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1556 m  CI95 [-1.3898,-1.0161] SIG  (ipm 2.008 -> geo 0.853)
 C09 dp90    = -1.4379 m  CI95 [-1.5511,-0.6778] SIG  (ipm 3.226 -> geo 1.788)
 C09 drmse   = -1.1464 m  CI95 [-1.3067,-0.9197] SIG  (ipm 2.278 -> geo 1.132)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.1778 m  CI95 [-1.3532,-0.9445] SIG  (ipm 1.909 -> geo 0.732)
 C10 dp90    = -1.2910 m  CI95 [-1.3770,-0.5720] SIG  (ipm 2.975 -> geo 1.684)
 C10 drmse   = -1.0493 m  CI95 [-1.1850,-0.8666] SIG  (ipm 2.133 -> geo 1.084)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.1373 m  CI95 [-1.2300,-1.0221] SIG  (ipm 1.931 -> geo 0.793)
 ALL dp90    = -1.2217 m  CI95 [-1.4578,-0.8353] SIG  (ipm 3.002 -> geo 1.781)
 ALL drmse   = -1.0427 m  CI95 [-1.1516,-0.9184] SIG  (ipm 2.165 -> geo 1.122)

## seed2 geo->final (C2 residual learning)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.0147 m  CI95 [-0.1906,+0.0608] ns  (geo 0.811 -> final 0.796)
 C08 dp90    = +0.0729 m  CI95 [-0.0317,+0.0792] ns  (geo 2.060 -> final 2.133)
 C08 drmse   = +0.0139 m  CI95 [-0.0349,+0.0499] ns  (geo 1.178 -> final 1.192)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -0.0055 m  CI95 [-0.0822,+0.0573] ns  (geo 0.853 -> final 0.847)
 C09 dp90    = -0.0378 m  CI95 [-0.2189,+0.0525] ns  (geo 1.788 -> final 1.750)
 C09 drmse   = -0.0426 m  CI95 [-0.0922,+0.0039] ns  (geo 1.132 -> final 1.089)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = +0.0608 m  CI95 [-0.0267,+0.1255] ns  (geo 0.732 -> final 0.792)
 C10 dp90    = -0.0391 m  CI95 [-0.0563,+0.1442] ns  (geo 1.684 -> final 1.645)
 C10 drmse   = +0.0065 m  CI95 [-0.0261,+0.0491] ns  (geo 1.084 -> final 1.091)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = +0.0217 m  CI95 [-0.0615,+0.0583] ns  (geo 0.793 -> final 0.815)
 ALL dp90    = -0.0329 m  CI95 [-0.1657,+0.0335] ns  (geo 1.781 -> final 1.748)
 ALL drmse   = -0.0114 m  CI95 [-0.0387,+0.0143] ns  (geo 1.122 -> final 1.111)

## seed2 ipm->final (total)
 C08 n_obs=11989 n_tracks=73
 C08 dmedian = -0.9919 m  CI95 [-1.2676,-0.6486] SIG  (ipm 1.788 -> final 0.796)
 C08 dp90    = -0.7274 m  CI95 [-0.9669,-0.6582] SIG  (ipm 2.861 -> final 2.133)
 C08 drmse   = -0.7944 m  CI95 [-0.9214,-0.6568] SIG  (ipm 1.986 -> final 1.192)
 C09 n_obs=24223 n_tracks=152
 C09 dmedian = -1.1611 m  CI95 [-1.4116,-1.0095] SIG  (ipm 2.008 -> final 0.847)
 C09 dp90    = -1.4757 m  CI95 [-1.5322,-0.8666] SIG  (ipm 3.226 -> final 1.750)
 C09 drmse   = -1.1890 m  CI95 [-1.3325,-0.9706] SIG  (ipm 2.278 -> final 1.089)
 C10 n_obs=24558 n_tracks=140
 C10 dmedian = -1.1170 m  CI95 [-1.3604,-0.9011] SIG  (ipm 1.909 -> final 0.792)
 C10 dp90    = -1.3300 m  CI95 [-1.3787,-0.5862] SIG  (ipm 2.975 -> final 1.645)
 C10 drmse   = -1.0428 m  CI95 [-1.2001,-0.8326] SIG  (ipm 2.133 -> final 1.091)
 ALL n_obs=60770 n_tracks=365
 ALL dmedian = -1.1156 m  CI95 [-1.2414,-0.9890] SIG  (ipm 1.931 -> final 0.815)
 ALL dp90    = -1.2546 m  CI95 [-1.4862,-0.9153] SIG  (ipm 3.002 -> final 1.748)
 ALL drmse   = -1.0541 m  CI95 [-1.1621,-0.9438] SIG  (ipm 2.165 -> final 1.111)

## identity metrics on test window (baseline geometric remerge vs seeds)
 gate     method     DetA     AssA     LocA     HOTA     IDF1     MOTA   IDsw
  0.5   baseline   0.0988   0.4040   0.4462   0.1998   0.1615  -0.9837    117
  0.5      seed0   0.0989   0.4479   0.4339   0.2105   0.1611  -0.9463     98
  0.5      seed1   0.0984   0.4390   0.4541   0.2079   0.1607  -0.9507    109
  0.5      seed2   0.1012   0.4510   0.4592   0.2136   0.1669  -0.9382     91
  1.0   baseline   0.2091   0.4553   0.5066   0.3086   0.2885  -0.5921    252
  1.0      seed0   0.1987   0.4499   0.5082   0.2990   0.2742  -0.5959    225
  1.0      seed1   0.2045   0.4415   0.5064   0.3005   0.2790  -0.5796    246
  1.0      seed2   0.2045   0.4486   0.5134   0.3029   0.2813  -0.5786    239
  1.5   baseline   0.2920   0.4914   0.5600   0.3788   0.3656  -0.3398    305
  1.5      seed0   0.2985   0.4890   0.5390   0.3820   0.3729  -0.2966    289
  1.5      seed1   0.3004   0.4852   0.5441   0.3818   0.3725  -0.2936    306
  1.5      seed2   0.2988   0.4876   0.5514   0.3817   0.3720  -0.2965    288
  2.0   baseline   0.3497   0.5378   0.6055   0.4336   0.4165  -0.1830    344
  2.0      seed0   0.3555   0.5367   0.5937   0.4368   0.4225  -0.1450    315
  2.0      seed1   0.3555   0.5349   0.5992   0.4361   0.4216  -0.1474    337
  2.0      seed2   0.3539   0.5371   0.6032   0.4360   0.4210  -0.1501    319

## BEV-HOTA (mean over 0.5/1/1.5/2 m gates)
  baseline  BEV-HOTA = 0.3302
     seed0  BEV-HOTA = 0.3321
     seed1  BEV-HOTA = 0.3316
     seed2  BEV-HOTA = 0.3336
