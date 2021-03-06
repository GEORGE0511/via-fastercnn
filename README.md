# via-fastercnn
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
cloned from https://github.com/akshaylamba/FasterRCNN_KERAS

Please note that I currently am quite busy with other projects and unfortunately dont have a lot of time to spend on this maintaining this repository, but any contributions are welcome!


USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.

- `train_frcnn.py` can be used to train a model. To train on Pascal VOC data, simply do:
  `python train_frcnn.py -p /path/to/pascalvoc/`. 

- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

- **simple_parser.py provides an alternative way to input data, using a Via tagged data http://www.robots.ox.ac.uk/~vgg/software/via/via.html.** 

  **You can directly mark by via, and train your own dataset.**

   **You don't need to use voc, coco format dataset.Simply provide a picture and path to json data,** 
    **use the command line option `-o simple`. For example `python train_frcnn.py -o simple -p /path/to/picture -j /path/to/json data`.**

- Running `train_frcnn.py` will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
  settings can then be loaded by `test_frcnn.py` for any testing.

- test_frcnn.py can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
  images:
    `python test_frcnn.py -p /path/to/test_data/`

- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical flips and `--rot` for 90 degree rotations



NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
- The theano backend by default uses a 7x7 pooling region, instead of 14x14 as in the frcnn paper. This cuts down compiling time slightly.
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is much more efficient and has little impact on results.

Example output:

![æ¥çæºå¾å](https://cn.bing.com/th?id=OIP.2HQjN367Tn-S4dvZfXgSAwHaFj&pid=Api&rs=1&p=0)

![preview](https://pic3.zhimg.com/v2-468921caa00e188e31ea8af23d9a999a_r.jpg)

ISSUES:

- If you get this error:
`ValueError: There is a negative shape in the graph!`    
    than update keras to the newest version

- This repo was developed using `python3` . python2`should work thanks to the contribution of a number of users.

- If you run out of memory, try reducing the number of ROIs that are processed simultaneously. Try passing a lower `-n` to `train_frcnn.py`. Alternatively, try reducing the image size from the default value of 600 (this setting is found in `config.py`.
