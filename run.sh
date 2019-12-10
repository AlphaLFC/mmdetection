echo "Start training and testing..."
python tools/train.py configs/fcos/fcos_plus_mstrain_r50_caffe_fpn_gn_fp16_1x_1gpu.py #--resume_from work_dirs/fcos_plus_mstrain_r50_caffe_fpn_gn_fp16_1x_1gpu/latest.pth
python tools/test.py configs/fcos/fcos_plus_mstrain_r50_caffe_fpn_gn_fp16_1x_1gpu.py  work_dirs/fcos_plus_mstrain_r50_caffe_fpn_gn_fp16_1x_1gpu/latest.pth --out work_dirs/fcos_plus_mstrain_r50_caffe_fpn_gn_fp16_1x_1gpu/test_results.pkl --eval bbox
echo "Done!"

