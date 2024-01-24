./configs/voc/vanilla.sh voc_id --resume checkpoint/checkpoint_voc_vanilla.pth --eval --viz_prediction_results
./configs/voc/vanilla.sh coco_ood --resume checkpoint/checkpoint_voc_vanilla.pth --eval --viz_prediction_results
./configs/voc/vanilla.sh openimages_ood --resume checkpoint/checkpoint_voc_vanilla.pth --eval --viz_prediction_results

./configs/bdd/vanilla.sh bdd_id --resume checkpoint/checkpoint_bdd_vanilla.pth --eval --viz_prediction_results
./configs/bdd/vanilla.sh coco_ood --resume checkpoint/checkpoint_bdd_vanilla.pth --eval --viz_prediction_results
./configs/bdd/vanilla.sh openimages_ood --resume checkpoint/checkpoint_bdd_vanilla.pth --eval --viz_prediction_results