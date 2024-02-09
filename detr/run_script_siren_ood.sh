./configs/voc/siren.sh voc_id --resume checkpoint/checkpoint_voc_siren.pth --eval 
./configs/voc/siren.sh voc_id --resume checkpoint/checkpoint_voc_siren.pth  --eval --maha_train 
./configs/voc/siren.sh coco_ood --resume checkpoint/checkpoint_voc_siren.pth --eval 
python voc_coco_vmf.py --name siren --pro_length 16 --use_trained_params 1