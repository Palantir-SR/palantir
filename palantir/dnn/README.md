### How to train a DNN? 
```
$PALANTIR_CODE_ROOT/palantir/dnn/script/train_video.sh -g [gup index] -c [content index] -q [quality level] -i [input resolution] -o [output resolution]
(e.g., $PALANTIR_CODE_ROOT/palantir/dnn/script/train_video.sh -g 0 -c 1 -q high -i 480 -o 2160)
```
* `content index`: It indicates the index of the content used in Palantir.
* A DNN is saved at `$PALANTIR_DATA_ROOT/[content index]/checkpoint/[video name]/[dnn name]`. (e.g., `$PALANTIR_DATA_ROOT/1/checkpoint/480p_1800kbps_s0_d300.webm/PALANTIR_S_B8_F48_S4_deconv/`)