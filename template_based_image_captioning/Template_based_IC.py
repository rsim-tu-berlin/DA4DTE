import pickle
from pickle import load
import json
import os

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile as tiff

import geopandas as gpd
import rasterio
from rasterio.plot import reshape_as_image
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
import pyproj
import warnings
warnings.filterwarnings('ignore')
from rasterio.transform import xy

from shapely.geometry import Point, shape



# Dictionary to convert numbers into words
num2words1 = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', \
             6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', \
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', \
            19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty', \
            50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', \
            90: 'ninety', 0: 'zero'}
num2words2 = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
num2words3 = ['hundred ']

# Function that convert numbers into words
def number(Number):
    if 0 <= Number <= 19:
        return num2words1[Number]
    elif 20 <= Number <= 99:
        tens, remainder = divmod(Number, 10)
        return num2words2[tens - 2] + '-' + num2words1[remainder] if remainder else num2words2[tens - 2]
    
    elif 100 <= Number <= 999:
        hundreds, remainder_1 = divmod(Number, 100)
        if 0 <= remainder_1 <= 19:
            return num2words1[hundreds] + ' ' + num2words3[0] + ' ' + num2words1[remainder_1]

        elif 20 <= remainder_1 <= 99:
            tens, remainder_10 = divmod(remainder_1, 10)


            return num2words1[hundreds] + ' ' + num2words3[0] + num2words2[tens - 2] + '-' + num2words1[remainder_10] if remainder_10 else num2words2[tens - 2]
    else:
        print('NaN')
        return 'NaN'
    
# BBX center coordinates
def centroid(coords):
    x, y = 0, 0
    n = len(coords)
    signed_area = 0
    for i in range(len(coords)):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    # print (signed_area)
    x /= 6 * signed_area
    y /= 6 * signed_area
    return (x, y)



def distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

# BBX surface calculation. This will be used to define the size of the ships.
def object_surface(coords):
    x, y = 0, 0
    n = len(coords)
    surface_objects = list()
    signed_area = 0
    for i in range(len(coords)):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    # print (signed_area)
    surface_objects.append(abs(signed_area))
    return surface_objects


# subdivide the image into 9 equal parts, starting from the top left;
# patch zero is the upper-left, 

# This function is used to dived images into patches to determine ships locations;
def divide_image_into_patches(image_width, image_height, num_patches=9):
    # Create a 128x128 grid
    # plt.figure(figsize=(8, 8))
    # plt.grid(True, linestyle='--', alpha=0.5)

    # patch_size = image_size / 3
    patch_width = image_width / 3
    patch_height = image_height / 3

    # Plot the grid and divide into patches
    for i in range(1, 3):
        plt.plot([i * patch_width, i * patch_width], [0, image_width], color='black', linestyle='-', linewidth=0.5)
        plt.plot([0, image_height], [i * patch_height, i * patch_height], color='black', linestyle='-', linewidth=0.5)

    # Draw rectangles to represent patches
    patches = {}
    patch_index = 0
    for i in range(3):
        for j in range(3):
            x = j * patch_width
            y = i * patch_height
            patch = Rectangle((x, y), patch_width, patch_height, edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
            # plt.gca().add_patch(patch)
            patches[patch_index] = {'x': x, 'y': y}
            patch_index += 1

    # Add labels and legend
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Image Divided into Patches')

    # Show the plot
    # plt.show()

    return patches

def get_patch_for_coordinate(image_width, image_height,coordinate, patches):
    # patch_size = image_size / 3
    patch_width = image_width / 3
    patch_height = image_height / 3
    for patch_index, patch_coordinates in patches.items():
        if (
            patch_coordinates['x'] <= coordinate[0] < patch_coordinates['x'] + patch_width and
            patch_coordinates['y'] <= coordinate[1] < patch_coordinates['y'] + patch_height
        ):
            return patch_index

    return None



# This function is used to convert degrees into meeters; This due to shp file which has a projection 'EPSG:4326';
# This will be used to calculate the distance between bbx of the ships and the nearst coastline point to
# determine whether a ships is in an harbor or near the coastline;
def degrees_to_meters(distance_deg, latitude):
    # Radius of the Earth at the given latitude (in meters)
    radius_earth = 6378137  # Earth's radius at the equator (in meters)
    
    # Convert latitude from degrees to radians
    latitude_rad = math.radians(latitude)
    
    # Calculate the circumference of the Earth at the given latitude
    circumference = 2 * math.pi * radius_earth * math.cos(latitude_rad)
    
    # Convert the distance in degrees to meters
    distance_meters = (distance_deg / 360) * circumference
    
    return distance_meters


# Determines the location of the ships in a given image;
def side(p_index):
    area_coordinates = {
        4: 'center',
        5: 'right',
        3: 'left',
        1: 'upper',
        7: 'lower',
        2: 'upper-right',
        0: 'upper-left',
        8: 'lower-right',
        6: 'lower-left',
    }
    return area_coordinates[p_index]


def replace_word(original_list, old_word, new_word):
    return [new_word if word == old_word else word for word in original_list]




number(95)
# path to the vessel detection dataset
path_data = '/mnt/storagecube/genc/data_vess/VesselDetection/v3/vessel_detection_dataset_v3/'

# path of the coastlione shape file
path_data_coastline = '/mnt/storagecube/genc/data_vess/'


# shp file read. Projection 'EPSG:4326'
track_line = gpd.read_file(path_data_coastline+'coastlines-split-4326/coastlines-split-4326/lines.shp')

# vessel bbox annotations
with open(path_data + 'vessels_dataset_annotations.json', encoding='utf-8') as data_file:
    dataset_annotation = json.loads(data_file.read())


annotaions_dict_t1 = {}
annotaions_dict_t1 = dataset_annotation
annotaions_dict_t1 == dataset_annotation
l= 0
no_files = 0
ids_list_keys = list()
template_1_list = list()
template_2_list = list()
template_3_list = list()
template_4_list = list()

spatial_index = track_line.sindex
track_line['geometry'] = track_line['geometry'].apply(lambda geom: geom if geom.geom_type == 'LineString' else LineString(geom))

# in the dataset some patches used to have zero pixel values; we created a json file with those patches names;
with open(path_data +'zero_patches.json', 'r') as zp:
    zero_patches = json.load(zp)

# here starts the template based image captioning;
l = 0
print ('Temlate base IC started:')
for k in annotaions_dict_t1:

    if os.path.exists (path_data +'/tif/' + k + '.tif') and k not in zero_patches['zero_patches'] :
        # print ('file tiff exists, file is equal:',k)
        # print ('ciao 2', k)

# We read each image to understand its coordinates
        with rasterio.open(path_data +'/tif/' + k + '.tif') as src:
            image = src.read()  # Read the image data
            image_bounds = src.bounds  # Get the image bounds
            transformation = src.transform
            raster_height = src.height

        image_height,image_width,  = int(image.shape[1]), int(image.shape[2])
        # l = l+1
        # if l==10:
        #     break

        n = len(annotaions_dict_t1[k]['annotations'])
        bboxes = annotaions_dict_t1[k]['annotations']
        coords_list = [l['geometry']['coordinates'][0] for l in bboxes]
        coords_list_flipped = list()

# The original images that were delivered by E-GEOS are flipped; We reflipp the bounding boxes
# to be correctly georeferenced;
        for geo in bboxes:
            coord = geo['geometry']['coordinates'][0]
            # print ('original',coord)
            adjusted_bounding_boxes = []
            for bbox in coord:
                # print ('bbox', bbox[0])
                adjusted_bounding_box = [bbox[0], raster_height - bbox[1]]  # Adjust y-coordinate
                adjusted_bounding_boxes.append(adjusted_bounding_box)
            coords_list_flipped.append(adjusted_bounding_boxes)
 
        area_lists = [object_surface(coords) for coords in coords_list]
        area_lists_flipped= [object_surface(coords) for coords in coords_list_flipped]
        area_lists_flipped = area_lists_flipped

        centroids_list = [centroid(coords) for coords in coords_list]
        centroids_list_flipped = [centroid(coords) for coords in coords_list_flipped]

        patches = divide_image_into_patches(image_width, image_height)

        patch_index = [get_patch_for_coordinate(image_width,image_height, coords,patches) for coords in centroids_list_flipped]

        list_size_ship = list()

        
        
        x1,y1,x2,y2 = image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3]

        # # Define the WGS84 geographic coordinate systems
        wgs84_utm_zone_33n = pyproj.Proj(init=(src.crs))  # EPSG:32633 (WGS 84 / UTM zone 33N)
        # print (wgs84_utm_zone_33n)
        wgs84_lat_lon_ESPG4326 = pyproj.Proj(init='EPSG:4326')  # EPSG:4326 (WGS 84)
        # print (wgs84_lat_lon_ESPG4326)
        # # distance calculate
        # wgs84_lat_lon_EPSG_3857 = pyproj.Proj(init='EPSG:3857') 

        # # Perform the transformation
        lon_1, lat_1 = pyproj.transform(wgs84_utm_zone_33n, wgs84_lat_lon_ESPG4326, x1, y1)
        lon_2, lat_2 = pyproj.transform(wgs84_utm_zone_33n, wgs84_lat_lon_ESPG4326, x2, y2)

        # # print("Longitude:", lon_1)
        # # print("Latitude:", lat_1)
        image_bounds = (lon_1, lat_1, lon_2, lat_2)  # (minx, miny, maxx, maxy)
        # # print ('image bounds',image_bounds)
        # # Create a bounding box polygon
        bbox_polygon = box(*image_bounds)
        

        # # Check if the bounding box intersects with the coastline
        intersects_coastline = track_line.intersects(bbox_polygon).any()#

        # Tempalte creations based on the number of vessels
        if n==1:
            template_2 = ''
            template_3 = 'mask_n mask_obj is present'
            template_1 = 'mask_n mask_obj'
            template_4 = 'mask_n mask_obj'
            # template_4 = ''

        if n!=1:
            template_2 = ' '
            template_3 = 'mask_n mask_obj are present'
            template_1 = 'mask_n mask_obj'
            template_4 = 'mask_n mask_obj'

        if n==0:
            template_2 = 'other types of land use land cover classes'
            template_3 = 'other types of land use land cover classes'
            template_1 = 'other types of land use land cover classes'
            template_4 = 'other types of land use land cover classes'


        # split the templates into tokens to substude the masked words with numbers, sizes and objects
        template_1_split = template_1.split()
        template_2_split = template_2.split()
        template_3_split = template_3.split()
        template_4_split = template_4.split()

        # if the image patch intersects the coastline do all the calculations to understand if in harbor or near the coast
        if intersects_coastline:

            nearest_index = list(spatial_index.intersection(bbox_polygon.bounds))[0]
            nearest_indexes = list(spatial_index.intersection(bbox_polygon.bounds))

            # if we have bbx (ships)
            if n!=0:

                bounding_box_georef = []
                bounding_box_georef_EPSG_4326 = list()
                distance_degree = list ()
                distances_index = []
                
                points_list = list()

                
                for n_boxes in range(len(coords_list_flipped)):
                    
                    coords_list_flipped[n_boxes].append([centroids_list_flipped[n_boxes][0],centroids_list_flipped[n_boxes][1]])
                    # print ('b',coords_list_flipped)
                    for points in coords_list_flipped[n_boxes]:

                        # N.B Center coordiantes it works. first element is y while the second element is x

                        x_c, y_c = xy(transformation, points[1], points[0])


                        bounding_box_georef.append([x_c,y_c])
                        xb_0,yb_0 = pyproj.transform(wgs84_utm_zone_33n, wgs84_lat_lon_ESPG4326, x_c,y_c)
                        # xb_1,yb_1 = pyproj.transform(wgs84_utm_zone_33n, wgs84_lat_lon_ESPG4326, x_c,y_c)
                        boxbox = (xb_0,yb_0)
                        bounding_box_georef_EPSG_4326.append(boxbox)
                        # distances = list()
                        distances = list()

                        for indx in nearest_indexes:
                            nearest_geometry = track_line.geometry.iloc[indx]
                            point = Point(boxbox)
                            distance = point.distance(nearest_geometry)
                            distances.append(distance)
                            points_list.append(point)


                        distances_index.extend(distances)

                min_distance = min(distances_index)
                max_distance = max(distances_index)
                # print ('min_distance', min_distance)
                # print ('distances',distances_index)
                np_dist = np.asarray(distances_index)
                idx_min = np.argmin(np_dist)
                idx_max = np.argmax(np_dist)
                mins_dist_degr = np_dist[idx_min]
                max_dist_degr = np_dist[idx_max]

                # print (mins_dist_degr)
                mins_lat = points_list[idx_min]

              
                distance_meters = degrees_to_meters(mins_dist_degr, mins_lat.y)
                distance_meters_max =  degrees_to_meters(max_dist_degr, mins_lat.y)
                # print("Minimum distance in meters:", distance_meters)
                # print("Maximum distance in meters:", distance_meters_max)
                # if one of the ships has a distance <10 meters from the coast then we assume that the ships are found in the harbor
                if distance_meters<10:
                    number_word = number(n)
                    number_word_3= number(n)
                    if n ==1:
                        object_word = 'vessel'

                    else:

                        object_word = 'vessels'

                        if n>5:
                            number_word_3 = 'many'
                            number_word ='many'

                    where_object_4 =  'in' ,'the', 'harbor'
                    where_object_3 =  'in' ,'the', 'port'
                    template_3_split = replace_word(template_3_split,'mask_n', number_word_3)
                    template_3_split = replace_word(template_3_split,'mask_obj', object_word)
                    
                    template_3_split[:0]=  where_object_3 
                    # print (template_3_split[:0])


                    template_4_split = replace_word(template_4_split,'mask_n', number_word)
                    template_4_split = replace_word(template_4_split,'mask_obj', object_word)
                    template_4_split.extend(where_object_4)
                    # print (template_4_split)
                 
                else:
                    number_word = number(n)
                    number_word_3= number(n)
                    if n ==1:
                        object_word = 'vessel'
                    else:
                        object_word = 'vessels'
                        if n>5:
                            number_word_3 = 'many'
                            number_word ='many'

                    where_object_4 =  'near' ,'the', 'coast'
                    where_object_3 =  'near' ,'the', 'seacoast'
                    template_4_split = replace_word(template_4_split,'mask_n', number_word)
                    template_4_split = replace_word(template_4_split,'mask_obj', object_word)
                    template_4_split.extend(where_object_4)

                    template_3_split = replace_word(template_3_split,'mask_n', number_word_3)
                    template_3_split = replace_word(template_3_split,'mask_obj', object_word)
                    template_3_split[:0]=  where_object_3
                    # print (template_4_split)
                    # print ('A satellite image with ships in a coastline.')
            else:
                template_4 = 'other types of land use land cover classes'
                template_4_split = template_4.split()

        else:
            
            if n!= 0:
                number_word = number(n)
                number_word_3= number(n)
                if n ==1:
                    object_word = 'vessel'
                else:
                    object_word = 'vessels'
                    if n >5:
                        number_word_3 = 'many'
                        number_word = 'many'
                    
                template_4_split = replace_word(template_4_split,'mask_n', number_word)
                template_4_split = replace_word(template_4_split,'mask_obj', object_word)

                template_3_split = replace_word(template_3_split,'mask_n', number_word_3)
                template_3_split = replace_word(template_3_split,'mask_obj', object_word)

            else:
                template_4 = 'other types of land use land cover classes'
                template_4_split = template_4.split()

        # Here we determine the area (surface of the bbx) in terms of pixels
        for area in range(len(area_lists_flipped)):
            # print ('area', area)
            if area_lists[area][0] < 12:
                list_size_ship.append(0)
            elif 12<=area_lists[area][0]<24:
                list_size_ship.append(1)
            elif 24<=area_lists[area][0]<100:
                list_size_ship.append(2)
            elif 100<=area_lists[area][0]<450:
                list_size_ship.append(3)
            elif 450<=area_lists[area][0]:
                list_size_ship.append(4)
        
        #vs = very small, sm = small, med = medium-sized, big = big, vbig = very big 
        vs= list_size_ship.count(0)
        sm= list_size_ship.count(1)
        med=list_size_ship.count(2)
        big = list_size_ship.count(3)
        vbig = list_size_ship.count(4)

        if vs == 1:
            size_o = ['very','small']
            object_ = 'boat'
            template_2_split.append(number(vs))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if vs != 1 and vs != 0:
            size_o = ['very','small']
            object_ = 'boats'
            if vs <6:
                template_2_split.append(number(vs))
            else:
                template_2_split.append('many') 
            # template_2_split.append(number(vs))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if sm == 1 :
            if vs == 1 or vs != 1 and vs != 0:
                template_2_split.append(' and')
            # print ('sm == ',sm)
            size_o = ['small']
            object_ = 'boat'
            template_2_split.append(number(sm))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if sm != 1 and sm !=0: 
            if vs == 1 or vs != 1 and vs != 0:
                template_2_split.append(' and')
            size_o = ['small']
            object_ = 'boats'
            if sm <6:
                template_2_split.append(number(sm))
            else:
                template_2_split.append('many')            
            # template_2_split.append(number(sm))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if med ==1:
            if vs == 1 or vs != 1 and vs != 0 or sm == 1 or sm != 1 and sm != 0:
                template_2_split.append(' and')
            size_o = ['medium-sized']
            object_ = 'boat'
            template_2_split.append(number(med))
            template_2_split.extend(size_o)
            template_2_split.append(object_)
        if med !=1 and med !=0:
            if vs == 1 or vs != 1 and vs != 0 or sm == 1 or sm != 1 and sm != 0:
                template_2_split.append(' and')
            size_o = ['medium-sized']
            object_ = 'boats'
            if med <6:
                template_2_split.append(number(med))
            else:
                template_2_split.append('many')            

            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if big ==1:
            if vs == 1 or vs != 1 and vs != 0 or sm == 1 or sm != 1 and sm != 0 or med == 1 or med != 1 and med != 0:
                template_2_split.append(' and')
            size_o = ['big']
            object_ = 'ship'
            template_2_split.append(number(big))
            template_2_split.extend(size_o)
            template_2_split.append(object_)
        if big !=1 and big !=0:
            if vs == 1 or vs != 1 and vs != 0 or sm == 1 or sm != 1 and sm != 0 or med == 1 or med != 1 and med != 0:
                template_2_split.append(' and')
            size_o = ['big']
            object_ = 'ships'
            if big <6:
                template_2_split.append(number(big))
            else:
                template_2_split.append('many')
            template_2_split.append(number(big))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if vbig ==1:
            if vs == 1 or vs != 1 and vs != 0 or sm == 1 or sm != 1 and sm != 0 or med == 1 or med != 1 and med != 0 or big == 1 or big != 1 and big != 0:
                template_2_split.append(' and')
            size_o = ['very','big']
            object_ = 'ship'
            template_2_split.append(number(vbig))
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        if vbig !=1 and vbig !=0:
            #sm != 1 : #and sm != 0
            size_o = ['very','big']
            object_ = 'ships'
            if vbig <6:
                template_2_split.append(number(vbig))
            else:
                template_2_split.append('many')
            template_2_split.extend(size_o)
            template_2_split.append(object_)

        
        
        joined_template_2_split = ' '.join(template_2_split)
        # print (joined_template_2_split)
        joined_template_3_split = ' '.join(template_3_split)

        words_1 = joined_template_2_split.split(' and')
        # print (words_1)
        words_2 = joined_template_3_split.split(' and')

        lenth_words_2 = len(words_2)
        if lenth_words_2>2:

            words_2.insert(lenth_words_2-1,' and')
            new_sentence_2 = ' '.join(words_2)
            spl_2= new_sentence_2.split()
            joined_template_3_split =' '.join(spl_2)


        template_3_list.append(joined_template_3_split)

        lenth_words_1 = len(words_1)
        if lenth_words_1>2:
            words_1.insert(lenth_words_1-1,' and')
            new_sentence = ' '.join(words_1)
            
            spl= new_sentence.split()
            joined_template_2_split =' '.join(spl)

            

        template_2_list.append(joined_template_2_split)



        for i in template_1_split:
            if n == 1:
                number_word = number(n)
                object_word = 'vessel' 
                template_1_split = replace_word(template_1_split, 'mask_n', number_word)
                template_1_split = replace_word(template_1_split, 'mask_obj', object_word)

            elif 6>n>1:
                number_word = number(n)
                object_word = 'vessels' 
                template_1_split = replace_word(template_1_split, 'mask_n', number_word)
                template_1_split = replace_word(template_1_split, 'mask_obj', object_word)

            elif n>=6:
                number_word = 'many'
                object_word = 'vessels' 
                template_1_split = replace_word(template_1_split, 'mask_n', number_word)
                template_1_split = replace_word(template_1_split, 'mask_obj', object_word)
            
            else:
                template_1_split = "other types of land use land cover classes".split()
                
                

        joined_template_1_split = ' '.join(template_1_split)
        joined_template_4_split = ' '.join(template_4_split)
        template_4_list.append(joined_template_4_split)

        ids_list_keys.append(k)
        template_1_list.append(joined_template_1_split)


    else:
        no_files = no_files+1



# print (len(ids_list_keys ))
# print (len(template_1_list)) 
# print(len(template_2_list)) 
# print (len(template_3_list))
# print (len(template_4_list))

# print (template_1_list)
# print (template_2_list)
# print (template_3_list)
# print (template_4_list)


# create json files
vessel_detection = {}
sentence_dict = {}
vessel_detection['dataset'] = 'vessel_captioning'


vessel_detection['images'] = list()

l = 0
idsimg = 0
for i in range (len(ids_list_keys)):
    idsimg =idsimg+3
    a = {'filename': ids_list_keys[i],
    'sentences': [{'image_id': i,
    'sentence_id': idsimg-3,
    'raw': template_1_list[i],
    'tokens': template_1_list[i].split()},
    {'image_id': i,
    'sentence_id': idsimg-2,
    'raw': template_2_list[i],
    'tokens':  template_2_list[i].split()},
    {'image_id': i,
    'sentence_id': idsimg-1,
    'raw': template_3_list[i],
    'tokens':template_3_list[i].split()},
    {'image_id': i,
    'sentence_id': idsimg,
    'raw': template_4_list[i],
    'tokens': template_4_list[i].split()}],
    'sentids': [idsimg-3,idsimg-2, idsimg-1, idsimg]}
  
    idsimg= idsimg +1
    vessel_detection['images'].append(a)
    # if l == 5:
    #     break
# print (vessel_detection)  
path_data = '/mnt/storagecube/genc/data_vess/VesselDetection/v3/vessel_detection_dataset_v3/'
with open(path_data +'vessel_captioning_prova_1.json', 'w') as fp:
    json.dump(vessel_detection, fp) 
print ('Caption Generation ended')

