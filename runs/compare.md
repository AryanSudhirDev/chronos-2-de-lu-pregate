Window 2023-01-01..2025-10-31 (1035 days), DM reference 580124a4ca (lear).

| Model | id | MAE | RMSE | Battery % of oracle | DM p vs ref |
|---|---|---|---|---|---|
| blend_pool7 | fd73754ae2 | 9.86 | 19.03 | 95.1 | 0.0000 |
| blend_pool7 | d99e45c506 | 9.88 | 19.06 | 94.3 | 0.0000 |
| blend_pool3 | 2dae4f56b9 | 9.88 | 18.97 | 94.9 | 0.0000 |
| blend_pool7 | 9371338e11 | 10.04 | 19.43 | 95.1 | 0.0000 |
| lgbm_stack_lora | 3f75dbb0b1 | 10.19 | 19.10 | 94.6 | 0.0000 |
| chronos2_ft_lora | 51d3abf5b2 | 10.36 | 19.74 | 94.7 | 0.0000 |
| blend_pool7 | 2dbad3c104 | 10.44 | 20.15 | 95.0 | 0.0000 |
| chronos2_ft_full | dbd170cd58 | 10.46 | 19.54 | 94.1 | 0.0000 |
| chronos2_ft_full | f2f5ca7406 | 10.47 | 20.10 | 94.6 | 0.0000 |
| chronos2_ft_lora_multi | de49f09033 | 10.54 | 19.98 | 94.6 | 0.0000 |
| tabpfn2_stack_lora | 8cf7cb666a | 10.58 | 19.35 | 95.2 | 0.0000 |
| ensemble_lora_lgbm | 13f7720eea | 10.59 | 20.01 | 95.1 | 0.0000 |
| lgbm_stack_zs | 85313a733a | 10.61 | 20.34 | 94.7 | 0.0000 |
| ensemble_lora_lgbm_lear | f846a34128 | 10.69 | 19.98 | 94.5 | 0.0000 |
| ensemble_zs_lgbm | ea25dc5487 | 10.88 | 20.83 | 95.0 | 0.0000 |
| ensemble_zs_lgbm_lear | d9d89d4949 | 10.89 | 20.48 | 94.5 | 0.0000 |
| chronos2_zs | 7a7adf81e5 | 11.26 | 21.59 | 94.3 | 0.0000 |
| ensemble_zs_lear | 3b4d9631f6 | 11.29 | 20.95 | 93.4 | 0.0000 |
| chronos2_zs | 297007c63e | 11.38 | 21.43 | 94.3 | 0.0000 |
| chronos2_zs | 6fa0d05a99 | 11.53 | 21.88 | 94.1 | 0.0000 |
| chronos2_zs | 8324d03f4e | 11.61 | 21.84 | 93.9 | 0.0000 |
| lgbm_hl | 8133309c04 | 12.53 | 22.08 | 93.1 | 0.0018 |
| lgbm | 69b2219339 | 12.65 | 22.44 | 93.2 | 0.0090 |
| tabpfn2 | 8851fdfc81 | 12.68 | 21.95 | 94.5 | 0.0049 |
| lear | 580124a4ca | 13.28 | 22.97 | 90.0 |  |
| lgbm_hl | f10dc65157 | 14.31 | 24.42 | 92.4 | 0.9997 |
| chronos2_ft_lora_gate | d8d9083325 | 16.18 | 27.89 | 92.6 | 1.0000 |
| chronos2_zs | 4e71fd0d1a | 16.51 | 28.34 | 92.5 | 1.0000 |
| lgbm_stack_gate | 2ee96d5f3e | 16.98 | 28.63 | 91.4 | 1.0000 |
| lgbm_stack_gate_lora | 221a0547f8 | 17.02 | 28.49 | 91.9 | 1.0000 |
| chronos2_zs | 2c0716d255 | 17.80 | 30.21 | 91.8 | 1.0000 |
| lgbm_gate | 1f92d7dfe1 | 18.54 | 30.04 | 90.5 | 1.0000 |
| naive | 73450863f2 | 27.99 | 43.54 | 86.2 | 1.0000 |
| naive_weekly | 17ca79b952 | 33.26 | 51.31 | 84.4 | 1.0000 |
