import cv2
import numpy as np
import cv2
import os
import json
def get_data(input_path,json_path):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    classes_count['pore'] = 0
    class_mapping = {}	
    ann = json.load(open(json_path))
    print('Parsing annotation files')
    ann = ann['_via_img_metadata']
    for img_id, key in enumerate(ann.keys()):
        filename = ann[key]['filename']
        regions = ann[key]["regions"]
        class_name= 'pore'# make annotations info and sorage it in coco_output['annotations']
        classes_count['pore'] += 1
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            points_x = region['shape_attributes']['x']
            points_y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']
            x = points_x
            y = points_y
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                img = cv2.imread(os.path.join(input_path,filename))
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = os.path.join(input_path,filename)
                all_imgs[filename]['width'] = width
                all_imgs[filename]['height'] = height
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x), 'x2': int(x+width), 'y1': int(y), 'y2': int(y+height)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
    return all_data, classes_count, class_mapping
