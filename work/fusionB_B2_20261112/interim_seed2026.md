# 中间结果，不需要等回复
仅seed2026；数值为比例。协调者已纠正32次笔误：正式24次，此时完成8次，剩余2027/2028两轮。
|频段/指标|argmax R/Z/F/F5|thr_fa1 R/Z/F/F5|
|---|---|---|
|2.4GHz v3_test_FA|0.0250 / 0.0125 / 0.0250 / 0.0750|0.0250 / 0.0000 / 0.0000 / 0.0750|
|2.4GHz nonG_miss|0.0643 / 0.0790 / 0.0818 / 0.0368|0.0358 / 0.2077 / 0.1930 / 0.0276|
|2.4GHz v3_drone_recall|0.8190 / 0.7905 / 0.8286 / 0.8857|0.9143 / 0.4667 / 0.6762 / 0.9143|
|2.4GHz B_FA|0.0250 / 0.0125 / 0.0250 / 0.0750|0.0250 / 0.0000 / 0.0000 / 0.0750|
|2.4GHz B_nonG_recall|0.8190 / 0.7905 / 0.8286 / 0.8857|0.9143 / 0.4667 / 0.6762 / 0.9143|
|2.4GHz mini3pro|0.8125 / 0.9375 / 0.7500 / 0.8750|0.9375 / 0.3750 / 0.5000 / 0.9375|
|2.4GHz mavic2|0.6500 / 0.4500 / 0.6000 / 0.7000|0.8000 / 0.1500 / 0.3000 / 0.7500|
|2.4GHz bgA_test_FA|0.0000 / 0.0000 / 0.0000 / 0.0000|0.0000 / 0.0000 / 0.0000 / 0.0000|
|2.4GHz C1_FA|0.0379 / 0.0150 / 0.0157 / 0.0293|0.0721 / 0.0000 / 0.0007 / 0.0357|
|2.4GHz notched_noise_FA|0.0000 / 0.0000 / 0.0000 / 0.0000|0.0000 / 0.0000 / 0.0000 / 0.0000|
|5.8GHz v3_test_FA|0.0000 / 0.0000 / 0.0167 / 0.0000|0.0000 / 0.0167 / 0.0167 / 0.0000|
|5.8GHz nonG_miss|0.0153 / 0.0000 / 0.0102 / 0.0077|0.0064 / 0.0000 / 0.0089 / 0.0077|
|5.8GHz v3_drone_recall|0.8308 / 0.9462 / 0.8462 / 0.8615|0.9077 / 0.9538 / 0.8615 / 0.8692|
|5.8GHz B_FA|0.0000 / 0.0000 / 0.0167 / 0.0000|0.0000 / 0.0167 / 0.0167 / 0.0000|
|5.8GHz B_nonG_recall|0.8854 / 1.0000 / 0.9167 / 0.9375|0.9479 / 1.0000 / 0.9271 / 0.9375|
|5.8GHz mini3pro|0.4500 / 1.0000 / 0.6000 / 0.7000|0.7500 / 1.0000 / 0.6500 / 0.7000|
|5.8GHz mavic2|1.0000 / 1.0000 / 1.0000 / 1.0000|1.0000 / 1.0000 / 1.0000 / 1.0000|
2.4GHz F 单列打乱argmax召回变化前三：papr999_db=-0.0533；log_gamma_max=-0.0362；c42n=-0.0267
2.4GHz F 依赖度变化：argmax/shuffle_all/nonG_miss=-0.0219；argmax/shuffle_all/v3_drone_recall=0.0248；thr_fa1/shuffle_all/nonG_miss=-0.0184；thr_fa1/shuffle_all/v3_drone_recall=-0.0095；argmax/zero/nonG_miss=0.0239；argmax/zero/v3_drone_recall=-0.0571；thr_fa1/zero/nonG_miss=-0.0009；thr_fa1/zero/v3_drone_recall=-0.0381
2.4GHz F−Z：argmax/v3_test_FA=0.0125；argmax/nonG_miss=0.0028；argmax/v3_drone_recall=0.0381；argmax/C1_FA=0.0007；thr_fa1/v3_test_FA=0.0000；thr_fa1/nonG_miss=-0.0147；thr_fa1/v3_drone_recall=0.2095；thr_fa1/C1_FA=0.0007
2.4GHz F5 单列打乱argmax召回变化前三：duty=0.0114；papr999_db=0.0057；sigma_f=0.0019
2.4GHz F5 依赖度变化：argmax/shuffle_all/nonG_miss=-0.0006；argmax/shuffle_all/v3_drone_recall=0.0095；thr_fa1/shuffle_all/nonG_miss=0.0007；thr_fa1/shuffle_all/v3_drone_recall=-0.0019；argmax/zero/nonG_miss=-0.0074；argmax/zero/v3_drone_recall=0.0190；thr_fa1/zero/nonG_miss=-0.0074；thr_fa1/zero/v3_drone_recall=0.0000
2.4GHz F5−Z：argmax/v3_test_FA=0.0625；argmax/nonG_miss=-0.0423；argmax/v3_drone_recall=0.0952；argmax/C1_FA=0.0143；thr_fa1/v3_test_FA=0.0750；thr_fa1/nonG_miss=-0.1801；thr_fa1/v3_drone_recall=0.4476；thr_fa1/C1_FA=0.0357
5.8GHz F 单列打乱argmax召回变化前三：papr999_db=-0.0092；shape_factor=0.0077；c42n=-0.0077
5.8GHz F 依赖度变化：argmax/shuffle_all/nonG_miss=-0.0023；argmax/shuffle_all/v3_drone_recall=0.0231；thr_fa1/shuffle_all/nonG_miss=-0.0010；thr_fa1/shuffle_all/v3_drone_recall=0.0185；argmax/zero/nonG_miss=-0.0026；argmax/zero/v3_drone_recall=0.0462；thr_fa1/zero/nonG_miss=-0.0013；thr_fa1/zero/v3_drone_recall=0.0308
5.8GHz F−Z：argmax/v3_test_FA=0.0167；argmax/nonG_miss=0.0102；argmax/v3_drone_recall=-0.1000；thr_fa1/v3_test_FA=0.0000；thr_fa1/nonG_miss=0.0089；thr_fa1/v3_drone_recall=-0.0923
5.8GHz F5 单列打乱argmax召回变化前三：occ_frac=0.0215；papr999_db=0.0000；sigma_f=0.0000
5.8GHz F5 依赖度变化：argmax/shuffle_all/nonG_miss=-0.0036；argmax/shuffle_all/v3_drone_recall=0.0446；thr_fa1/shuffle_all/nonG_miss=-0.0056；thr_fa1/shuffle_all/v3_drone_recall=0.0615；argmax/zero/nonG_miss=-0.0064；argmax/zero/v3_drone_recall=0.0692；thr_fa1/zero/nonG_miss=-0.0077；thr_fa1/zero/v3_drone_recall=0.0692
5.8GHz F5−Z：argmax/v3_test_FA=0.0000；argmax/nonG_miss=0.0077；argmax/v3_drone_recall=-0.0846；thr_fa1/v3_test_FA=-0.0167；thr_fa1/nonG_miss=0.0077；thr_fa1/v3_drone_recall=-0.0846
单种子结果；逐片k/N与CP95在CSV中；三种子标准差及方向一致性待后两轮，不下结论。
