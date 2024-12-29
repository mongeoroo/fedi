LOG_DIR=logger
mkdir -p ${LOG_DIR}
echo ${LOG_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

script -c '{
  set -x
  echo 'script---------------------------------------------------------------------------'
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_pretrain_amp.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10501' --multiprocessing-distributed --world-size 1 --rank 0 \
  --logger pretrain_fedi --lr 2 --epochs 100 --pred-dim 8192 --batch-size 1024 --print-freq 100\
  -data data/imagenet
  echo 'script---------------------------------------------------------------------------'
  set +x
  }' ${LOG_DIR}/fedi-100ep-amp-${NOW}.log
