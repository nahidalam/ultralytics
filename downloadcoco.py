from ultralytics.utils.downloads import download
from pathlib import Path
import yaml

# Download labels
segments = True  # segment or box labels
# Assuming 'yaml_file_path' is the path to your YAML file
with open('coco.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)

# Assuming 'yaml_data' is a dictionary containing the path information
dirpath = Path(yaml_data['path'])  # dataset root dir
#dirpath = Path(yaml['/home/nahalam/dataset'])  # dataset root dir
url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
download(urls, dir=dirpath.parent)
# Download data
urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
          'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
          'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
download(urls, dir=dirpath /'images', threads=3)
