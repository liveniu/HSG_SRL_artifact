# lumpi_M6 supervised subgraph ['C05', 'C06'] (test split)

Paper map (fab3): Tables III--IV footnote (BEV-HOTA 32.9--34.1%).
Not Table VII (objective-ablation batch).

## seed0 ipm->geo (C1 forward geometry, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.3107 m  CI95 [-1.3616,-1.1849] SIG  (ipm 2.136 -> geo 0.825)
  dp90    = -1.3551 m  CI95 [-1.4450,-1.2475] SIG  (ipm 2.789 -> geo 1.434)
  drmse   = -1.2176 m  CI95 [-1.2808,-1.1412] SIG  (ipm 2.193 -> geo 0.976)

## seed0 geo->final (C2 residual learning, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -0.1671 m  CI95 [-0.2525,-0.1339] SIG  (geo 0.825 -> final 0.658)
  dp90    = -0.2152 m  CI95 [-0.2964,-0.1334] SIG  (geo 1.434 -> final 1.218)
  drmse   = -0.1544 m  CI95 [-0.1880,-0.1224] SIG  (geo 0.976 -> final 0.821)

## seed0 ipm->final (total, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.4778 m  CI95 [-1.5681,-1.3535] SIG  (ipm 2.136 -> final 0.658)
  dp90    = -1.5704 m  CI95 [-1.6426,-1.4821] SIG  (ipm 2.789 -> final 1.218)
  drmse   = -1.3720 m  CI95 [-1.4469,-1.2898] SIG  (ipm 2.193 -> final 0.821)

## seed1 ipm->geo (C1 forward geometry, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.3107 m  CI95 [-1.3670,-1.1956] SIG  (ipm 2.136 -> geo 0.825)
  dp90    = -1.3551 m  CI95 [-1.4532,-1.2377] SIG  (ipm 2.789 -> geo 1.434)
  drmse   = -1.2176 m  CI95 [-1.2788,-1.1462] SIG  (ipm 2.193 -> geo 0.976)

## seed1 geo->final (C2 residual learning, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -0.2175 m  CI95 [-0.3095,-0.1772] SIG  (geo 0.825 -> final 0.608)
  dp90    = -0.2163 m  CI95 [-0.3112,-0.1453] SIG  (geo 1.434 -> final 1.217)
  drmse   = -0.1868 m  CI95 [-0.2164,-0.1524] SIG  (geo 0.976 -> final 0.789)

## seed1 ipm->final (total, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.5283 m  CI95 [-1.6136,-1.3967] SIG  (ipm 2.136 -> final 0.608)
  dp90    = -1.5714 m  CI95 [-1.7018,-1.4741] SIG  (ipm 2.789 -> final 1.217)
  drmse   = -1.4044 m  CI95 [-1.4817,-1.3088] SIG  (ipm 2.193 -> final 0.789)

## seed2 ipm->geo (C1 forward geometry, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.3107 m  CI95 [-1.3641,-1.1857] SIG  (ipm 2.136 -> geo 0.825)
  dp90    = -1.3551 m  CI95 [-1.4438,-1.2387] SIG  (ipm 2.789 -> geo 1.434)
  drmse   = -1.2176 m  CI95 [-1.2837,-1.1478] SIG  (ipm 2.193 -> geo 0.976)

## seed2 geo->final (C2 residual learning, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -0.2029 m  CI95 [-0.2906,-0.1583] SIG  (geo 0.825 -> final 0.622)
  dp90    = -0.2636 m  CI95 [-0.3545,-0.1587] SIG  (geo 1.434 -> final 1.170)
  drmse   = -0.1855 m  CI95 [-0.2224,-0.1453] SIG  (geo 0.976 -> final 0.790)

## seed2 ipm->final (total, subgraph pooled)
  n_obs=49487 n_tracks=498
  dmedian = -1.5136 m  CI95 [-1.5996,-1.3894] SIG  (ipm 2.136 -> final 0.622)
  dp90    = -1.6188 m  CI95 [-1.7697,-1.4858] SIG  (ipm 2.789 -> final 1.170)
  drmse   = -1.4031 m  CI95 [-1.4888,-1.3087] SIG  (ipm 2.193 -> final 0.790)

## identity metrics (subgraph only)
 gate     method     DetA     AssA     LocA     HOTA     IDF1     MOTA   IDsw
  0.5   baseline   0.0753   0.2085   0.4053   0.1253   0.1122  -0.3123    419
  0.5      seed0   0.0995   0.2571   0.4208   0.1599   0.1474  -0.2480    513
  0.5      seed1   0.1185   0.3296   0.3971   0.1977   0.1734  -0.2049    589
  0.5      seed2   0.1161   0.3234   0.3812   0.1938   0.1717  -0.2066    506
  1.0   baseline   0.2027   0.3363   0.4210   0.2611   0.2484  -0.0269    890
  1.0      seed0   0.2650   0.4692   0.4608   0.3526   0.3334   0.0977    999
  1.0      seed1   0.2799   0.4817   0.4822   0.3672   0.3439   0.1226   1067
  1.0      seed2   0.2825   0.4814   0.4712   0.3688   0.3480   0.1304   1003
  1.5   baseline   0.3154   0.4381   0.4969   0.3718   0.3512   0.1827   1117
  1.5      seed0   0.3405   0.4652   0.5629   0.3980   0.3825   0.2283   1135
  1.5      seed1   0.3405   0.4626   0.5889   0.3969   0.3786   0.2259   1190
  1.5      seed2   0.3409   0.4647   0.5865   0.3980   0.3814   0.2292   1136
  2.0   baseline   0.3470   0.4343   0.5913   0.3882   0.3700   0.2352   1172
  2.0      seed0   0.3544   0.4624   0.6571   0.4048   0.3889   0.2499   1192
  2.0      seed1   0.3532   0.4603   0.6773   0.4032   0.3854   0.2459   1233
  2.0      seed2   0.3542   0.4625   0.6754   0.4047   0.3891   0.2503   1172

## BEV-HOTA (mean over 0.5/1/1.5/2 m gates)
  baseline  BEV-HOTA = 0.2866
     seed0  BEV-HOTA = 0.3288
     seed1  BEV-HOTA = 0.3412
     seed2  BEV-HOTA = 0.3413
