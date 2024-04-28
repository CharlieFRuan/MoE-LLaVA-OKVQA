Authors: Ankit Gupta, Charlie Ruan, Qiru Hu, Rohan Chawla, Sarah Di (equal contribution)

This work is based on https://github.com/PKU-YuanGroup/MoE-LLaVA. To finetune on OK-VQA, run `./scripts/v1/stablelm/finetune_moe_approach2.sh`.

To evaluate the result, run `python eval_script_approach2.py`.

For the final OK-VQA score, run `approaches_eval.ipynb` (need previous two steps).

Various setups are needed, e.g. setting up the environment following https://github.com/PKU-YuanGroup/MoE-LLaVA cloning the original MoE-LLaVA weights `MoE-LLaVA-StableLM-1.6B-4e`, preparing the OKVQA data, etc.
