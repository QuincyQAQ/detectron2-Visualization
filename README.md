### `visualization.py` README

#### Overview
The `visualization.py` script is designed to visualize various types of data, including images, bounding boxes, and scores associated with those bounding boxes. This script leverages several classes and utilities defined in other files to achieve its functionality.


#### Usage
```python
python visualization.py
```

### `visualization_d2toYolo.py` README

#### Overview
The `visualization_d2toYolo.py` script is designed to convert and visualize data from a Detectron2 format to the YOLO format. This script is particularly useful for scenarios where you need to visualize ground-truth data or annotations in a YOLO-compatible format. 



#### Usage
To use the `visualization_d2toYolo.py` script, you need to run it from the command line with the appropriate arguments.

```bash
python visualization_d2toYolo.py --source annotation --config-file path_to_config.yaml --output-dir ./output --show
```

This command will visualize the annotations in the YOLO format and save the output to the specified directory. If the `--show` flag is provided, the output will also be displayed in a window. If you need to change the category displayed on the detection box, modify data/datasets/builtin_meta.py

#### Dependencies
- `argparse`: For parsing command-line arguments.
- `cv2`: OpenCV library for image processing and visualization.
- `numpy`: For numerical operations on arrays.

#### Notes
- Ensure that the configuration file (`--config-file`) is correctly formatted and contains the necessary parameters for the conversion and visualization process.
- The script assumes that the input data is in Detectron2 format and will convert it to YOLO format for visualization.

This README provides a brief overview of the functionality and usage of the `visualization.py` and `visualization_d2toYolo.py` scripts. For more detailed information, refer to the source code and comments within the files.
