# lumpi_M6 step4 three-layer evaluation (test split)

## seed0 ipm->geo (C1 forward geometry)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.0358 m  CI95 [-1.1553,-0.8607] SIG  (ipm 1.802 -> geo 0.766)
 C05 dp90    = -1.2206 m  CI95 [-1.3749,-1.0802] SIG  (ipm 2.825 -> geo 1.605)
 C05 drmse   = -0.9701 m  CI95 [-1.0732,-0.8257] SIG  (ipm 2.012 -> geo 1.042)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.3441 m  CI95 [-1.4344,-1.2591] SIG  (ipm 2.202 -> geo 0.858)
 C06 dp90    = -1.4909 m  CI95 [-1.5581,-1.4008] SIG  (ipm 2.750 -> geo 1.260)
 C06 drmse   = -1.3483 m  CI95 [-1.4032,-1.2829] SIG  (ipm 2.286 -> geo 0.938)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6718,-1.5535] SIG  (ipm 2.236 -> geo 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6056,-1.5283] SIG  (ipm 2.884 -> geo 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5237,-1.3917] SIG  (ipm 2.317 -> geo 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.4096 m  CI95 [-1.4738,-1.3613] SIG  (ipm 2.168 -> geo 0.758)
 ALL dp90    = -1.4838 m  CI95 [-1.5410,-1.3873] SIG  (ipm 2.849 -> geo 1.365)
 ALL drmse   = -1.3110 m  CI95 [-1.3603,-1.2573] SIG  (ipm 2.242 -> geo 0.931)

## seed0 geo->final (C2 residual learning)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -0.0503 m  CI95 [-0.1222,+0.0135] ns  (geo 0.766 -> final 0.716)
 C05 dp90    = -0.2879 m  CI95 [-0.3488,-0.1473] SIG  (geo 1.605 -> final 1.317)
 C05 drmse   = -0.0920 m  CI95 [-0.1482,-0.0380] SIG  (geo 1.042 -> final 0.950)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -0.2375 m  CI95 [-0.2954,-0.2028] SIG  (geo 0.858 -> final 0.621)
 C06 dp90    = -0.1970 m  CI95 [-0.2332,-0.1469] SIG  (geo 1.260 -> final 1.063)
 C06 drmse   = -0.1958 m  CI95 [-0.2275,-0.1611] SIG  (geo 0.938 -> final 0.742)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.625 -> final 0.625)
 C07 dp90    = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 1.313 -> final 1.313)
 C07 drmse   = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.857 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -0.1123 m  CI95 [-0.1642,-0.0574] SIG  (geo 0.758 -> final 0.646)
 ALL dp90    = -0.0968 m  CI95 [-0.1756,-0.0485] SIG  (geo 1.365 -> final 1.268)
 ALL drmse   = -0.0959 m  CI95 [-0.1212,-0.0695] SIG  (geo 0.931 -> final 0.835)

## seed0 ipm->final (total)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.0861 m  CI95 [-1.2425,-0.9223] SIG  (ipm 1.802 -> final 0.716)
 C05 dp90    = -1.5085 m  CI95 [-1.6069,-1.3824] SIG  (ipm 2.825 -> final 1.317)
 C05 drmse   = -1.0621 m  CI95 [-1.1834,-0.9497] SIG  (ipm 2.012 -> final 0.950)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.5816 m  CI95 [-1.7112,-1.5023] SIG  (ipm 2.202 -> final 0.621)
 C06 dp90    = -1.6879 m  CI95 [-1.7755,-1.5904] SIG  (ipm 2.750 -> final 1.063)
 C06 drmse   = -1.5441 m  CI95 [-1.6210,-1.4663] SIG  (ipm 2.286 -> final 0.742)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6708,-1.5552] SIG  (ipm 2.236 -> final 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6126,-1.5317] SIG  (ipm 2.884 -> final 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5224,-1.3896] SIG  (ipm 2.317 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.5219 m  CI95 [-1.5847,-1.4555] SIG  (ipm 2.168 -> final 0.646)
 ALL dp90    = -1.5806 m  CI95 [-1.6129,-1.5351] SIG  (ipm 2.849 -> final 1.268)
 ALL drmse   = -1.4069 m  CI95 [-1.4583,-1.3441] SIG  (ipm 2.242 -> final 0.835)

## seed1 ipm->geo (C1 forward geometry)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.0358 m  CI95 [-1.1550,-0.8476] SIG  (ipm 1.802 -> geo 0.766)
 C05 dp90    = -1.2206 m  CI95 [-1.3781,-1.0960] SIG  (ipm 2.825 -> geo 1.605)
 C05 drmse   = -0.9701 m  CI95 [-1.0738,-0.8257] SIG  (ipm 2.012 -> geo 1.042)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.3441 m  CI95 [-1.4376,-1.2660] SIG  (ipm 2.202 -> geo 0.858)
 C06 dp90    = -1.4909 m  CI95 [-1.5580,-1.4045] SIG  (ipm 2.750 -> geo 1.260)
 C06 drmse   = -1.3483 m  CI95 [-1.4037,-1.2814] SIG  (ipm 2.286 -> geo 0.938)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6685,-1.5564] SIG  (ipm 2.236 -> geo 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6095,-1.5265] SIG  (ipm 2.884 -> geo 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5232,-1.3937] SIG  (ipm 2.317 -> geo 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.4096 m  CI95 [-1.4712,-1.3675] SIG  (ipm 2.168 -> geo 0.758)
 ALL dp90    = -1.4838 m  CI95 [-1.5384,-1.3871] SIG  (ipm 2.849 -> geo 1.365)
 ALL drmse   = -1.3110 m  CI95 [-1.3610,-1.2593] SIG  (ipm 2.242 -> geo 0.931)

## seed1 geo->final (C2 residual learning)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -0.1139 m  CI95 [-0.2043,-0.0352] SIG  (geo 0.766 -> final 0.652)
 C05 dp90    = -0.2907 m  CI95 [-0.3375,-0.1848] SIG  (geo 1.605 -> final 1.314)
 C05 drmse   = -0.1270 m  CI95 [-0.1860,-0.0697] SIG  (geo 1.042 -> final 0.915)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -0.2765 m  CI95 [-0.3488,-0.2404] SIG  (geo 0.858 -> final 0.582)
 C06 dp90    = -0.2458 m  CI95 [-0.2866,-0.1868] SIG  (geo 1.260 -> final 1.014)
 C06 drmse   = -0.2268 m  CI95 [-0.2616,-0.1905] SIG  (geo 0.938 -> final 0.711)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.625 -> final 0.625)
 C07 dp90    = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 1.313 -> final 1.313)
 C07 drmse   = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.857 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -0.1455 m  CI95 [-0.2007,-0.0851] SIG  (geo 0.758 -> final 0.613)
 ALL dp90    = -0.1030 m  CI95 [-0.1761,-0.0547] SIG  (geo 1.365 -> final 1.262)
 ALL drmse   = -0.1152 m  CI95 [-0.1453,-0.0874] SIG  (geo 0.931 -> final 0.816)

## seed1 ipm->final (total)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.1497 m  CI95 [-1.3123,-0.9758] SIG  (ipm 1.802 -> final 0.652)
 C05 dp90    = -1.5113 m  CI95 [-1.6617,-1.3672] SIG  (ipm 2.825 -> final 1.314)
 C05 drmse   = -1.0971 m  CI95 [-1.2209,-0.9655] SIG  (ipm 2.012 -> final 0.915)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.6206 m  CI95 [-1.7807,-1.5498] SIG  (ipm 2.202 -> final 0.582)
 C06 dp90    = -1.7367 m  CI95 [-1.8124,-1.6335] SIG  (ipm 2.750 -> final 1.014)
 C06 drmse   = -1.5751 m  CI95 [-1.6553,-1.4910] SIG  (ipm 2.286 -> final 0.711)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6691,-1.5488] SIG  (ipm 2.236 -> final 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6076,-1.5225] SIG  (ipm 2.884 -> final 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5231,-1.3819] SIG  (ipm 2.317 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.5550 m  CI95 [-1.6189,-1.4900] SIG  (ipm 2.168 -> final 0.613)
 ALL dp90    = -1.5868 m  CI95 [-1.6260,-1.5364] SIG  (ipm 2.849 -> final 1.262)
 ALL drmse   = -1.4262 m  CI95 [-1.4777,-1.3650] SIG  (ipm 2.242 -> final 0.816)

## seed2 ipm->geo (C1 forward geometry)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.0358 m  CI95 [-1.1512,-0.8546] SIG  (ipm 1.802 -> geo 0.766)
 C05 dp90    = -1.2206 m  CI95 [-1.3723,-1.0802] SIG  (ipm 2.825 -> geo 1.605)
 C05 drmse   = -0.9701 m  CI95 [-1.0721,-0.8288] SIG  (ipm 2.012 -> geo 1.042)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.3441 m  CI95 [-1.4345,-1.2650] SIG  (ipm 2.202 -> geo 0.858)
 C06 dp90    = -1.4909 m  CI95 [-1.5597,-1.3962] SIG  (ipm 2.750 -> geo 1.260)
 C06 drmse   = -1.3483 m  CI95 [-1.4029,-1.2861] SIG  (ipm 2.286 -> geo 0.938)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6694,-1.5517] SIG  (ipm 2.236 -> geo 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6069,-1.5259] SIG  (ipm 2.884 -> geo 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5194,-1.3919] SIG  (ipm 2.317 -> geo 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.4096 m  CI95 [-1.4780,-1.3631] SIG  (ipm 2.168 -> geo 0.758)
 ALL dp90    = -1.4838 m  CI95 [-1.5401,-1.3904] SIG  (ipm 2.849 -> geo 1.365)
 ALL drmse   = -1.3110 m  CI95 [-1.3616,-1.2602] SIG  (ipm 2.242 -> geo 0.931)

## seed2 geo->final (C2 residual learning)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -0.0495 m  CI95 [-0.1018,+0.0030] ns  (geo 0.766 -> final 0.717)
 C05 dp90    = -0.2164 m  CI95 [-0.2525,-0.1246] SIG  (geo 1.605 -> final 1.388)
 C05 drmse   = -0.0724 m  CI95 [-0.1193,-0.0262] SIG  (geo 1.042 -> final 0.970)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -0.2722 m  CI95 [-0.3365,-0.2406] SIG  (geo 0.858 -> final 0.586)
 C06 dp90    = -0.3589 m  CI95 [-0.5326,-0.2776] SIG  (geo 1.260 -> final 0.901)
 C06 drmse   = -0.2649 m  CI95 [-0.3063,-0.2132] SIG  (geo 0.938 -> final 0.673)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.625 -> final 0.625)
 C07 dp90    = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 1.313 -> final 1.313)
 C07 drmse   = +0.0000 m  CI95 [+0.0000,+0.0000] ns  (geo 0.857 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -0.1351 m  CI95 [-0.1844,-0.0801] SIG  (geo 0.758 -> final 0.623)
 ALL dp90    = -0.1202 m  CI95 [-0.2153,-0.0553] SIG  (geo 1.365 -> final 1.245)
 ALL drmse   = -0.1144 m  CI95 [-0.1468,-0.0824] SIG  (geo 0.931 -> final 0.817)

## seed2 ipm->final (total)
 C05 n_obs=17444 n_tracks=311
 C05 dmedian = -1.0854 m  CI95 [-1.2254,-0.9122] SIG  (ipm 1.802 -> final 0.717)
 C05 dp90    = -1.4371 m  CI95 [-1.5635,-1.3092] SIG  (ipm 2.825 -> final 1.388)
 C05 drmse   = -1.0424 m  CI95 [-1.1548,-0.9163] SIG  (ipm 2.012 -> final 0.970)
 C06 n_obs=32043 n_tracks=187
 C06 dmedian = -1.6163 m  CI95 [-1.7583,-1.5388] SIG  (ipm 2.202 -> final 0.586)
 C06 dp90    = -1.8499 m  CI95 [-1.9836,-1.7305] SIG  (ipm 2.750 -> final 0.901)
 C06 drmse   = -1.6132 m  CI95 [-1.6822,-1.5267] SIG  (ipm 2.286 -> final 0.673)
 C07 n_obs=31579 n_tracks=256
 C07 dmedian = -1.6106 m  CI95 [-1.6679,-1.5525] SIG  (ipm 2.236 -> final 0.625)
 C07 dp90    = -1.5703 m  CI95 [-1.6032,-1.5277] SIG  (ipm 2.884 -> final 1.313)
 C07 drmse   = -1.4600 m  CI95 [-1.5228,-1.3948] SIG  (ipm 2.317 -> final 0.857)
 ALL n_obs=81066 n_tracks=754
 ALL dmedian = -1.5447 m  CI95 [-1.6077,-1.4774] SIG  (ipm 2.168 -> final 0.623)
 ALL dp90    = -1.6040 m  CI95 [-1.6634,-1.5368] SIG  (ipm 2.849 -> final 1.245)
 ALL drmse   = -1.4254 m  CI95 [-1.4841,-1.3623] SIG  (ipm 2.242 -> final 0.817)

## identity metrics on test window (baseline geometric remerge vs seeds)
 gate     method     DetA     AssA     LocA     HOTA     IDF1     MOTA   IDsw
  0.5   baseline   0.1006   0.2862   0.4500   0.1697   0.1510  -0.2387    673
  0.5      seed0   0.1294   0.3567   0.4481   0.2148   0.1953  -0.1452    699
  0.5      seed1   0.1270   0.3577   0.4597   0.2131   0.1888  -0.1588    756
  0.5      seed2   0.1318   0.3534   0.4450   0.2158   0.1972  -0.1416    721
  1.0   baseline   0.2112   0.3457   0.4973   0.2702   0.2545  -0.0016   1288
  1.0      seed0   0.2395   0.4099   0.5310   0.3133   0.3029   0.0751   1255
  1.0      seed1   0.2468   0.4165   0.5311   0.3206   0.3062   0.0825   1304
  1.0      seed2   0.2495   0.4149   0.5325   0.3218   0.3105   0.0924   1281
  1.5   baseline   0.2751   0.3680   0.5768   0.3182   0.3038   0.1153   1668
  1.5      seed0   0.2961   0.4086   0.6116   0.3478   0.3377   0.1753   1421
  1.5      seed1   0.2944   0.4094   0.6218   0.3472   0.3332   0.1667   1465
  1.5      seed2   0.2954   0.4080   0.6268   0.3472   0.3345   0.1723   1457
  2.0   baseline   0.2960   0.3673   0.6542   0.3297   0.3143   0.1521   1723
  2.0      seed0   0.3072   0.4084   0.6931   0.3542   0.3419   0.1939   1457
  2.0      seed1   0.3046   0.4097   0.7018   0.3533   0.3374   0.1838   1498
  2.0      seed2   0.3065   0.4078   0.7045   0.3536   0.3404   0.1910   1491

## BEV-HOTA (mean over 0.5/1/1.5/2 m gates)
  baseline  BEV-HOTA = 0.2719
     seed0  BEV-HOTA = 0.3076
     seed1  BEV-HOTA = 0.3085
     seed2  BEV-HOTA = 0.3096
