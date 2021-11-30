celeb=$1
exp_name=$2
checkpoints_dir=$3

python renderer/create_inputs.py --celeb $celeb --exp_name $exp_name --save_shapes
python renderer/test.py --celeb $celeb --exp_name $exp_name --checkpoints_dir $checkpoints_dir --which_epoch 20
python postprocessing/unalign.py --celeb $celeb --exp_name $exp_name
python postprocessing/blend.py --celeb $celeb --exp_name $exp_name --save_images
