### How to downlaod a video? 
```
$PALANTIR_CODE_ROOT/palantir/tool/script/download_video.sh -c [content index]
(e.g.,  $PALANTIR_CODE_ROOT/palantir/tool/script/download_video.sh -c 1)
```
* `content index`: It indicates the index of the content used in Palantir.
* The video's URL is provided at `$PALANTIR_CODE_ROOT/palantir/tool/video.py`.
* The video is downloaded at `$PALANTIR_DATA_ROOT/video/`.

### How to transcode a video?
```
$PALANTIR_CODE_ROOT/palantir/tool/script/encode_video.sh -c [content index] -g [gop]
(e.g.,  $PALANTIR_CODE_ROOT/palantir/tool/script/encode_video.sh -c 1 -g 60)
```
* The transcoded video clips (2160p and 480p) are stored at `$PALANTIR_DATA_ROOT/[content name]/video` (e.g., `$PALANTIR_DATA_ROOT/1/video`), with the gop (group of pictures) being 60.
