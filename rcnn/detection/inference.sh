seed=4400

python apply_net.py --dataset-dir ~/huy.nq/ood/data/VOC_0712_converted --test-dataset voc_custom_train --config-file VOC-Detection/faster-rcnn/center64_0.1.yaml --inference-config Inference/standard_nms.yaml --random-seed $seed --image-corruption-level 0 --visualize 1

python apply_net.py --dataset-dir ~/huy.nq/ood/data/VOC_0712_converted --test-dataset voc_custom_val --config-file VOC-Detection/faster-rcnn/center64_0.1.yaml --inference-config Inference/standard_nms.yaml --random-seed $seed --image-corruption-level 0 --visualize 1

python apply_net.py --dataset-dir ~/huy.nq/ood/data/COCO --test-dataset coco_ood_val  --config-file VOC-Detection/faster-rcnn/center64_0.1.yaml  --inference-config Inference/standard_nms.yaml --random-seed $seed --image-corruption-level 0 --visualize 1

python apply_net.py --dataset-dir ~/huy.nq/ood/data/OpenImages/ --test-dataset openimages_ood_val  --config-file VOC-Detection/faster-rcnn/center64_0.1.yaml  --inference-config Inference/standard_nms.yaml --random-seed $seed --image-corruption-level 0 --visualize 1

python voc_coco_knn.py --name center64_0.1 --seed $seed --ood coco
python voc_coco_knn.py --name center64_0.1 --seed $seed --ood openimages