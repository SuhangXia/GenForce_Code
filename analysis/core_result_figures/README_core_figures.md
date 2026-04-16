# Core Result Figures

## Files

- `fig1_hardest_test_main_metrics.(png|pdf)`: Hardest test main errors. Shows DWL-TR leads on x/y and overall MAE, while SCCWM leads on depth.
- `fig2_hardest_test_confound_metrics.(png|pdf)`: Hardest test confound-sensitive metrics. Shows SCCWM has higher CCAUC, DWL-TR slightly lower SASS.
- `fig3_heldout_val_main_metrics.(png|pdf)`: Held-out validation main errors. Shows DWL-TR is slightly better on x/y, SCCWM is better on depth and overall MAE.
- `fig4_heldout_val_confound_metrics.(png|pdf)`: Held-out validation confound-sensitive metrics. Shows SCCWM has higher CCAUC and similar SASS.
- `fig5_key_takeaways.(png|pdf)`: PPT-ready conclusion panel. Highlights SCCWM depth error reduction of 52.9% on hardest test and 33.1% on held-out val, plus stronger CCAUC.
- `fig6_summary_table.(png|pdf)`: Compact summary table with best values highlighted.

## Recommended Main Result Slide

- Use `fig5_key_takeaways.png` as the primary supervisor presentation slide.
- Use `fig6_summary_table.png` as the backup summary slide if a compact comparison table is needed.

## Data Sources

- `sccwm_hard`: `/home/suhang/datasets/checkpoints/sccwm_120h_full/eval_stage2_test_unseen_indenters_heldout_scales_limit40k.json`
- `dwl_hard`: `/home/suhang/datasets/checkpoints/dwl_tr_120h/eval_test_unseen_indenters_heldout_scales_limit40k.json`
- `sccwm_val`: `/home/suhang/datasets/checkpoints/sccwm_120h_full/eval_stage2_val_heldout_scale_bands_limit40k.json`
- `dwl_val`: `/home/suhang/datasets/checkpoints/dwl_tr_120h/eval_val_heldout_scale_bands_limit40k.json`
