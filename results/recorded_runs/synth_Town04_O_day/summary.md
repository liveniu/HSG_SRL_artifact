# synth_Town04_O_day step4 three-layer evaluation (test split)

## seed0 ipm->geo (C1 forward geometry)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.3856 m  CI95 [-1.2304,+0.9973] ns  (ipm 1.400 -> geo 1.014)
 C01 dp90    = -0.0154 m  CI95 [-0.2436,+0.5793] ns  (ipm 2.027 -> geo 2.011)
 C01 drmse   = -0.0064 m  CI95 [-0.6091,+0.5847] ns  (ipm 1.434 -> geo 1.427)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.6236 m  CI95 [-0.7303,+0.6076] ns  (ipm 1.687 -> geo 1.063)
 C02 dp90    = +0.3040 m  CI95 [-0.4986,+0.7252] ns  (ipm 2.016 -> geo 2.320)
 C02 drmse   = -0.1555 m  CI95 [-0.3457,+0.3589] ns  (ipm 1.837 -> geo 1.681)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2810 m  CI95 [-1.3737,-1.0432] SIG  (ipm 1.866 -> geo 0.585)
 C03 dp90    = -0.9035 m  CI95 [-2.6295,-0.6860] SIG  (ipm 2.372 -> geo 1.468)
 C03 drmse   = -1.1651 m  CI95 [-1.5626,-0.7903] SIG  (ipm 2.155 -> geo 0.990)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.2451 m  CI95 [-1.7258,-0.6147] SIG  (ipm 1.644 -> geo 0.399)
 C04 dp90    = -0.3530 m  CI95 [-1.3863,+0.3607] ns  (ipm 2.046 -> geo 1.693)
 C04 drmse   = -0.8118 m  CI95 [-1.2956,-0.3070] SIG  (ipm 1.702 -> geo 0.890)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8888 m  CI95 [-1.2521,-0.3666] SIG  (ipm 1.632 -> geo 0.744)
 ALL dp90    = -0.1341 m  CI95 [-0.2921,+0.2017] ns  (ipm 2.055 -> geo 1.920)
 ALL drmse   = -0.4324 m  CI95 [-0.7573,-0.0888] SIG  (ipm 1.701 -> geo 1.268)

## seed0 geo->final (C2 residual learning)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.0808 m  CI95 [-0.2124,+0.0027] ns  (geo 1.014 -> final 0.933)
 C01 dp90    = -0.1561 m  CI95 [-0.2261,-0.1009] SIG  (geo 2.011 -> final 1.855)
 C01 drmse   = -0.1465 m  CI95 [-0.1810,-0.0927] SIG  (geo 1.427 -> final 1.281)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = +0.0970 m  CI95 [-0.0229,+0.1722] ns  (geo 1.063 -> final 1.160)
 C02 dp90    = +0.1727 m  CI95 [+0.1239,+0.1860] SIG  (geo 2.320 -> final 2.493)
 C02 drmse   = +0.0938 m  CI95 [+0.0781,+0.1321] SIG  (geo 1.681 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = +0.0218 m  CI95 [-0.0224,+0.0858] ns  (geo 0.585 -> final 0.607)
 C03 dp90    = -0.0264 m  CI95 [-0.0680,+0.1069] ns  (geo 1.468 -> final 1.442)
 C03 drmse   = +0.0181 m  CI95 [-0.0151,+0.0518] ns  (geo 0.990 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = +0.0976 m  CI95 [-0.0443,+0.1544] ns  (geo 0.399 -> final 0.496)
 C04 dp90    = +0.1308 m  CI95 [-0.0356,+0.1593] ns  (geo 1.693 -> final 1.824)
 C04 drmse   = +0.0351 m  CI95 [-0.0022,+0.0730] ns  (geo 0.890 -> final 0.926)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.0006 m  CI95 [-0.0884,+0.0900] ns  (geo 0.744 -> final 0.743)
 ALL dp90    = +0.0174 m  CI95 [-0.2031,+0.0794] ns  (geo 1.920 -> final 1.938)
 ALL drmse   = -0.0334 m  CI95 [-0.0870,+0.0207] ns  (geo 1.268 -> final 1.235)

## seed0 ipm->final (total)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.4663 m  CI95 [-1.3635,+0.7854] ns  (ipm 1.400 -> final 0.933)
 C01 dp90    = -0.1716 m  CI95 [-0.4931,+0.3810] ns  (ipm 2.027 -> final 1.855)
 C01 drmse   = -0.1529 m  CI95 [-0.7730,+0.3903] ns  (ipm 1.434 -> final 1.281)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.5267 m  CI95 [-0.6101,+0.6754] ns  (ipm 1.687 -> final 1.160)
 C02 dp90    = +0.4767 m  CI95 [-0.3343,+0.8544] ns  (ipm 2.016 -> final 2.493)
 C02 drmse   = -0.0617 m  CI95 [-0.2567,+0.4666] ns  (ipm 1.837 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2592 m  CI95 [-1.3445,-1.0085] SIG  (ipm 1.866 -> final 0.607)
 C03 dp90    = -0.9299 m  CI95 [-2.5351,-0.7103] SIG  (ipm 2.372 -> final 1.442)
 C03 drmse   = -1.1470 m  CI95 [-1.5135,-0.7701] SIG  (ipm 2.155 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.1475 m  CI95 [-1.7247,-0.6161] SIG  (ipm 1.644 -> final 0.496)
 C04 dp90    = -0.2222 m  CI95 [-1.3378,+0.3680] ns  (ipm 2.046 -> final 1.824)
 C04 drmse   = -0.7767 m  CI95 [-1.2405,-0.2566] SIG  (ipm 1.702 -> final 0.926)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8893 m  CI95 [-1.2553,-0.3439] SIG  (ipm 1.632 -> final 0.743)
 ALL dp90    = -0.1167 m  CI95 [-0.3986,+0.1525] ns  (ipm 2.055 -> final 1.938)
 ALL drmse   = -0.4659 m  CI95 [-0.7949,-0.1806] SIG  (ipm 1.701 -> final 1.235)

## seed1 ipm->geo (C1 forward geometry)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.3856 m  CI95 [-1.2827,+0.9982] ns  (ipm 1.400 -> geo 1.014)
 C01 dp90    = -0.0154 m  CI95 [-0.2481,+0.5920] ns  (ipm 2.027 -> geo 2.011)
 C01 drmse   = -0.0064 m  CI95 [-0.6737,+0.5781] ns  (ipm 1.434 -> geo 1.427)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.6236 m  CI95 [-0.7219,+0.5863] ns  (ipm 1.687 -> geo 1.063)
 C02 dp90    = +0.3040 m  CI95 [-0.4371,+0.6803] ns  (ipm 2.016 -> geo 2.320)
 C02 drmse   = -0.1555 m  CI95 [-0.3405,+0.3413] ns  (ipm 1.837 -> geo 1.681)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2810 m  CI95 [-1.3714,-1.0243] SIG  (ipm 1.866 -> geo 0.585)
 C03 dp90    = -0.9035 m  CI95 [-2.6365,-0.6852] SIG  (ipm 2.372 -> geo 1.468)
 C03 drmse   = -1.1651 m  CI95 [-1.5823,-0.7611] SIG  (ipm 2.155 -> geo 0.990)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.2451 m  CI95 [-1.7457,-0.6153] SIG  (ipm 1.644 -> geo 0.399)
 C04 dp90    = -0.3530 m  CI95 [-1.3911,+0.3890] ns  (ipm 2.046 -> geo 1.693)
 C04 drmse   = -0.8118 m  CI95 [-1.2833,-0.3089] SIG  (ipm 1.702 -> geo 0.890)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8888 m  CI95 [-1.2226,-0.3613] SIG  (ipm 1.632 -> geo 0.744)
 ALL dp90    = -0.1341 m  CI95 [-0.3399,+0.2054] ns  (ipm 2.055 -> geo 1.920)
 ALL drmse   = -0.4324 m  CI95 [-0.7964,-0.0945] SIG  (ipm 1.701 -> geo 1.268)

## seed1 geo->final (C2 residual learning)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.0764 m  CI95 [-0.2132,+0.0066] ns  (geo 1.014 -> final 0.938)
 C01 dp90    = -0.1610 m  CI95 [-0.2259,-0.1073] SIG  (geo 2.011 -> final 1.850)
 C01 drmse   = -0.1473 m  CI95 [-0.1815,-0.0943] SIG  (geo 1.427 -> final 1.280)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = +0.0970 m  CI95 [-0.0168,+0.1711] ns  (geo 1.063 -> final 1.160)
 C02 dp90    = +0.1727 m  CI95 [+0.1024,+0.1860] SIG  (geo 2.320 -> final 2.493)
 C02 drmse   = +0.0938 m  CI95 [+0.0777,+0.1330] SIG  (geo 1.681 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = +0.0218 m  CI95 [-0.0323,+0.0866] ns  (geo 0.585 -> final 0.607)
 C03 dp90    = -0.0264 m  CI95 [-0.0630,+0.1068] ns  (geo 1.468 -> final 1.442)
 C03 drmse   = +0.0181 m  CI95 [-0.0116,+0.0510] ns  (geo 0.990 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = +0.0952 m  CI95 [-0.0282,+0.1492] ns  (geo 0.399 -> final 0.494)
 C04 dp90    = +0.0186 m  CI95 [-0.0884,+0.1382] ns  (geo 1.693 -> final 1.712)
 C04 drmse   = +0.0142 m  CI95 [-0.0247,+0.0621] ns  (geo 0.890 -> final 0.905)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.0039 m  CI95 [-0.0911,+0.0899] ns  (geo 0.744 -> final 0.740)
 ALL dp90    = +0.0049 m  CI95 [-0.2051,+0.0738] ns  (geo 1.920 -> final 1.925)
 ALL drmse   = -0.0386 m  CI95 [-0.0895,+0.0110] ns  (geo 1.268 -> final 1.230)

## seed1 ipm->final (total)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.4619 m  CI95 [-1.4290,+0.7826] ns  (ipm 1.400 -> final 0.938)
 C01 dp90    = -0.1764 m  CI95 [-0.4635,+0.3942] ns  (ipm 2.027 -> final 1.850)
 C01 drmse   = -0.1538 m  CI95 [-0.7667,+0.3766] ns  (ipm 1.434 -> final 1.280)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.5267 m  CI95 [-0.6463,+0.7801] ns  (ipm 1.687 -> final 1.160)
 C02 dp90    = +0.4767 m  CI95 [-0.2572,+0.9117] ns  (ipm 2.016 -> final 2.493)
 C02 drmse   = -0.0617 m  CI95 [-0.2756,+0.5328] ns  (ipm 1.837 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2592 m  CI95 [-1.3520,-0.9944] SIG  (ipm 1.866 -> final 0.607)
 C03 dp90    = -0.9299 m  CI95 [-2.5373,-0.7085] SIG  (ipm 2.372 -> final 1.442)
 C03 drmse   = -1.1470 m  CI95 [-1.5367,-0.8046] SIG  (ipm 2.155 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.1500 m  CI95 [-1.7235,-0.5391] SIG  (ipm 1.644 -> final 0.494)
 C04 dp90    = -0.3344 m  CI95 [-1.3587,+0.2815] ns  (ipm 2.046 -> final 1.712)
 C04 drmse   = -0.7976 m  CI95 [-1.2423,-0.2837] SIG  (ipm 1.702 -> final 0.905)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8926 m  CI95 [-1.2395,-0.4073] SIG  (ipm 1.632 -> final 0.740)
 ALL dp90    = -0.1292 m  CI95 [-0.4163,+0.1038] ns  (ipm 2.055 -> final 1.925)
 ALL drmse   = -0.4710 m  CI95 [-0.7631,-0.2130] SIG  (ipm 1.701 -> final 1.230)

## seed2 ipm->geo (C1 forward geometry)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.3856 m  CI95 [-1.1922,+0.9908] ns  (ipm 1.400 -> geo 1.014)
 C01 dp90    = -0.0154 m  CI95 [-0.3067,+0.5628] ns  (ipm 2.027 -> geo 2.011)
 C01 drmse   = -0.0064 m  CI95 [-0.6879,+0.5246] ns  (ipm 1.434 -> geo 1.427)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.6236 m  CI95 [-0.6648,+0.5843] ns  (ipm 1.687 -> geo 1.063)
 C02 dp90    = +0.3040 m  CI95 [-0.4707,+0.7196] ns  (ipm 2.016 -> geo 2.320)
 C02 drmse   = -0.1555 m  CI95 [-0.3419,+0.3435] ns  (ipm 1.837 -> geo 1.681)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2810 m  CI95 [-1.3697,-1.0049] SIG  (ipm 1.866 -> geo 0.585)
 C03 dp90    = -0.9035 m  CI95 [-2.6333,-0.6805] SIG  (ipm 2.372 -> geo 1.468)
 C03 drmse   = -1.1651 m  CI95 [-1.5701,-0.7645] SIG  (ipm 2.155 -> geo 0.990)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.2451 m  CI95 [-1.7277,-0.5936] SIG  (ipm 1.644 -> geo 0.399)
 C04 dp90    = -0.3530 m  CI95 [-1.3689,+0.3659] ns  (ipm 2.046 -> geo 1.693)
 C04 drmse   = -0.8118 m  CI95 [-1.2853,-0.2966] SIG  (ipm 1.702 -> geo 0.890)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8888 m  CI95 [-1.2345,-0.3699] SIG  (ipm 1.632 -> geo 0.744)
 ALL dp90    = -0.1341 m  CI95 [-0.3014,+0.2055] ns  (ipm 2.055 -> geo 1.920)
 ALL drmse   = -0.4324 m  CI95 [-0.7620,-0.0786] SIG  (ipm 1.701 -> geo 1.268)

## seed2 geo->final (C2 residual learning)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.0695 m  CI95 [-0.2136,+0.0227] ns  (geo 1.014 -> final 0.945)
 C01 dp90    = -0.1334 m  CI95 [-0.2260,-0.0789] SIG  (geo 2.011 -> final 1.878)
 C01 drmse   = -0.1468 m  CI95 [-0.1820,-0.0933] SIG  (geo 1.427 -> final 1.280)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = +0.0970 m  CI95 [-0.0230,+0.1792] ns  (geo 1.063 -> final 1.160)
 C02 dp90    = +0.1727 m  CI95 [+0.0995,+0.1860] SIG  (geo 2.320 -> final 2.493)
 C02 drmse   = +0.0938 m  CI95 [+0.0777,+0.1292] SIG  (geo 1.681 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = +0.0218 m  CI95 [-0.0396,+0.0857] ns  (geo 0.585 -> final 0.607)
 C03 dp90    = -0.0264 m  CI95 [-0.0645,+0.1069] ns  (geo 1.468 -> final 1.442)
 C03 drmse   = +0.0181 m  CI95 [-0.0147,+0.0524] ns  (geo 0.990 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = +0.1025 m  CI95 [-0.0430,+0.1578] ns  (geo 0.399 -> final 0.501)
 C04 dp90    = +0.1143 m  CI95 [-0.0378,+0.1578] ns  (geo 1.693 -> final 1.807)
 C04 drmse   = +0.0398 m  CI95 [-0.0041,+0.0791] ns  (geo 0.890 -> final 0.930)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = +0.0058 m  CI95 [-0.0863,+0.0934] ns  (geo 0.744 -> final 0.749)
 ALL dp90    = +0.0109 m  CI95 [-0.2054,+0.0748] ns  (geo 1.920 -> final 1.931)
 ALL drmse   = -0.0325 m  CI95 [-0.0866,+0.0179] ns  (geo 1.268 -> final 1.236)

## seed2 ipm->final (total)
 C01 n_obs=1605 n_tracks=22
 C01 dmedian = -0.4551 m  CI95 [-1.4320,+0.7840] ns  (ipm 1.400 -> final 0.945)
 C01 dp90    = -0.1489 m  CI95 [-0.4808,+0.3840] ns  (ipm 2.027 -> final 1.878)
 C01 drmse   = -0.1533 m  CI95 [-0.7908,+0.3749] ns  (ipm 1.434 -> final 1.280)
 C02 n_obs=596 n_tracks=13
 C02 dmedian = -0.5267 m  CI95 [-0.6349,+0.7016] ns  (ipm 1.687 -> final 1.160)
 C02 dp90    = +0.4767 m  CI95 [-0.4480,+0.8707] ns  (ipm 2.016 -> final 2.493)
 C02 drmse   = -0.0617 m  CI95 [-0.2725,+0.4712] ns  (ipm 1.837 -> final 1.775)
 C03 n_obs=600 n_tracks=18
 C03 dmedian = -1.2592 m  CI95 [-1.3481,-1.0034] SIG  (ipm 1.866 -> final 0.607)
 C03 dp90    = -0.9299 m  CI95 [-2.5398,-0.6991] SIG  (ipm 2.372 -> final 1.442)
 C03 drmse   = -1.1470 m  CI95 [-1.5408,-0.7779] SIG  (ipm 2.155 -> final 1.008)
 C04 n_obs=1270 n_tracks=15
 C04 dmedian = -1.1427 m  CI95 [-1.6904,-0.4872] SIG  (ipm 1.644 -> final 0.501)
 C04 dp90    = -0.2387 m  CI95 [-1.3249,+0.3373] ns  (ipm 2.046 -> final 1.807)
 C04 drmse   = -0.7720 m  CI95 [-1.2278,-0.1961] SIG  (ipm 1.702 -> final 0.930)
 ALL n_obs=4071 n_tracks=68
 ALL dmedian = -0.8830 m  CI95 [-1.2273,-0.3124] SIG  (ipm 1.632 -> final 0.749)
 ALL dp90    = -0.1231 m  CI95 [-0.4086,+0.1459] ns  (ipm 2.055 -> final 1.931)
 ALL drmse   = -0.4649 m  CI95 [-0.7569,-0.1481] SIG  (ipm 1.701 -> final 1.236)

## identity metrics on test window (baseline geometric remerge vs seeds)
 gate     method     DetA     AssA     LocA     HOTA     IDF1     MOTA   IDsw
  0.5   baseline   0.0992   0.3935   0.3159   0.1976   0.1713  -0.2340     19
  0.5      seed0   0.0974   0.3373   0.4055   0.1812   0.1702  -0.2397     17
  0.5      seed1   0.1038   0.3529   0.3950   0.1914   0.1797  -0.2114     18
  0.5      seed2   0.0935   0.3282   0.4278   0.1752   0.1624  -0.2370     19
  1.0   baseline   0.1786   0.4213   0.4934   0.2743   0.2710  -0.0525     36
  1.0      seed0   0.1715   0.4280   0.5385   0.2709   0.2672  -0.0692     36
  1.0      seed1   0.1737   0.4277   0.5399   0.2726   0.2688  -0.0540     40
  1.0      seed2   0.1733   0.4280   0.5362   0.2723   0.2678  -0.0552     42
  1.5   baseline   0.2381   0.4595   0.5716   0.3307   0.3413   0.0684     47
  1.5      seed0   0.2344   0.4485   0.5823   0.3242   0.3232   0.0608     41
  1.5      seed1   0.2391   0.4470   0.5803   0.3269   0.3267   0.0788     46
  1.5      seed2   0.2378   0.4477   0.5794   0.3263   0.3251   0.0761     48
  2.0   baseline   0.2842   0.4297   0.6065   0.3495   0.3649   0.1509     78
  2.0      seed0   0.2861   0.4197   0.6045   0.3465   0.3526   0.1547     69
  2.0      seed1   0.2908   0.4192   0.6061   0.3491   0.3578   0.1709     74
  2.0      seed2   0.2901   0.4194   0.6031   0.3488   0.3576   0.1693     76

## BEV-HOTA (mean over 0.5/1/1.5/2 m gates)
  baseline  BEV-HOTA = 0.2880
     seed0  BEV-HOTA = 0.2807
     seed1  BEV-HOTA = 0.2850
     seed2  BEV-HOTA = 0.2806
