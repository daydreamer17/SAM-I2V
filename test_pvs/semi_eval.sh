################## SAM-I2V ##################

# setup prompt
#input="3c"
#input="bb"
input="gm"
ckpt="sam-i2v_8gpu"
save_dir_name="davis_run1"   # 你推理用的

workers=16
prediction_name="Semi_SAVTest_${ckpt}_${input}"

python ../tools/sav_evaluator.py \
  --gt_root /root/autodl-tmp/DAVIS/Annotations_obj \
  --pred_root ./output_semi/${save_dir_name}/${prediction_name} \
  --num_processes ${workers} \
  --do_not_skip_first_and_last_frame

