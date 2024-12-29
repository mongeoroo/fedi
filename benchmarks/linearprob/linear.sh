LOG_DIR=logger
mkdir -p ${LOG_DIR}
echo ${LOG_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

script -c 'python benchmarks/linearprob/main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10051' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained pretrain_fedi/checkpoint_0399.pth.tar \
  --lars \
  data/imagenet' ${LOG_DIR}/linear-fedi-400ep-${NOW}.log 