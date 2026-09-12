# lumpi_M6 temporal extrapolation (train split of 0-360s, eval t>360s)

Paper map (fab3 Table VI):
- C06 dMedian -0.21 to -0.25 m is Table VI row 1 (later C06 interval).
- C07 in this file is a temporal force-all diagnostic (+0.08 to +0.15 m).
  It is not Table VI last row. That row is test-split force-all,
  +0.11 to +0.17 m. Principal Frozen C07 keeps DeltaP=0 (`summary.md`).

## seed0
 C05 n_obs=268 n_tracks=12
 C05 dmedian = +0.0915 m  CI95 [-0.1039,+0.1816] ns  (geo 0.703 -> final 0.794)
 C05 dp90    = -0.0873 m  CI95 [-0.3231,+0.3056] ns  (geo 1.269 -> final 1.182)
 C05 drmse   = +0.0373 m  CI95 [-0.0867,+0.1289] ns  (geo 0.881 -> final 0.918)
 C06 n_obs=116317 n_tracks=486
 C06 dmedian = -0.2097 m  CI95 [-0.2587,-0.1198] SIG  (geo 1.053 -> final 0.844)
 C06 dp90    = +0.0337 m  CI95 [-0.0145,+0.0813] ns  (geo 2.835 -> final 2.869)
 C06 drmse   = -0.0775 m  CI95 [-0.0950,-0.0556] SIG  (geo 2.195 -> final 2.118)
 C07 n_obs=227903 n_tracks=1828
 C07 dmedian = +0.0802 m  CI95 [+0.0553,+0.1048] SIG  (geo 0.619 -> final 0.700)
 C07 dp90    = +0.2050 m  CI95 [+0.1784,+0.2285] SIG  (geo 1.308 -> final 1.513)
 C07 drmse   = +0.1126 m  CI95 [+0.1007,+0.1243] SIG  (geo 0.882 -> final 0.995)

## seed1
 C05 n_obs=268 n_tracks=12
 C05 dmedian = +0.0658 m  CI95 [-0.2346,+0.1967] ns  (geo 0.703 -> final 0.769)
 C05 dp90    = -0.0397 m  CI95 [-0.3813,+0.3389] ns  (geo 1.269 -> final 1.230)
 C05 drmse   = +0.0404 m  CI95 [-0.1097,+0.1620] ns  (geo 0.881 -> final 0.921)
 C06 n_obs=116317 n_tracks=486
 C06 dmedian = -0.2521 m  CI95 [-0.3080,-0.1421] SIG  (geo 1.053 -> final 0.801)
 C06 dp90    = +0.0082 m  CI95 [-0.0385,+0.0496] ns  (geo 2.835 -> final 2.843)
 C06 drmse   = -0.0946 m  CI95 [-0.1131,-0.0747] SIG  (geo 2.195 -> final 2.101)
 C07 n_obs=227903 n_tracks=1828
 C07 dmedian = +0.0917 m  CI95 [+0.0724,+0.1097] SIG  (geo 0.619 -> final 0.711)
 C07 dp90    = +0.2355 m  CI95 [+0.2048,+0.2665] SIG  (geo 1.308 -> final 1.544)
 C07 drmse   = +0.1266 m  CI95 [+0.1122,+0.1404] SIG  (geo 0.882 -> final 1.009)

## seed2
 C05 n_obs=268 n_tracks=12
 C05 dmedian = +0.0837 m  CI95 [-0.1539,+0.1490] ns  (geo 0.703 -> final 0.786)
 C05 dp90    = -0.0584 m  CI95 [-0.3331,+0.1580] ns  (geo 1.269 -> final 1.211)
 C05 drmse   = +0.0121 m  CI95 [-0.1226,+0.0980] ns  (geo 0.881 -> final 0.893)
 C06 n_obs=116317 n_tracks=486
 C06 dmedian = -0.2537 m  CI95 [-0.3102,-0.1448] SIG  (geo 1.053 -> final 0.799)
 C06 dp90    = +0.0139 m  CI95 [-0.0286,+0.0647] ns  (geo 2.835 -> final 2.849)
 C06 drmse   = -0.0909 m  CI95 [-0.1096,-0.0686] SIG  (geo 2.195 -> final 2.104)
 C07 n_obs=227903 n_tracks=1828
 C07 dmedian = +0.1470 m  CI95 [+0.1231,+0.1689] SIG  (geo 0.619 -> final 0.766)
 C07 dp90    = +0.3655 m  CI95 [+0.3291,+0.4053] SIG  (geo 1.308 -> final 1.674)
 C07 drmse   = +0.2053 m  CI95 [+0.1833,+0.2275] SIG  (geo 0.882 -> final 1.087)

