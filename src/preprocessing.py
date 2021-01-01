import glob
import os
import functools
import csv
from collections import namedtuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
        'data-unversioned')

# candidate nodules which have been annotated as benign or malignant
# additionally we include the diameter of the nodule to achieve a comparable
# distribution of differently sized nodules in training and validation set
CandidateNoduleTuple = namedtuple(
    'CandidateNoduleTuple',
    'isNodule, diameter_mm, series_uid, center_xyz'
)


def getDownloadedSeriesUIDs(data_dir):
    return set([os.path.splitext(os.path.basename(p))[0] \
        for p in glob.glob(os.path.join(data_dir, 'subset*', '*.mhd'))])


def mapSeriesUIDsToAnnotations(annotations_path):
    annotation_dict = {}
    with open(annotations_path, 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center = tuple([float(x) for x in row[1:4]])
            diameter = float(row[4])
            annotation_dict.setdefault(series_uid, []).append(
                    (annotation_center, diameter))
    return annotation_dict


def too_far_apart(p1: tuple, p2: tuple, threshold: float):
    for i in range(len(p1)):
        if abs(p1[i] - p2[i]) > threshold:
            return True
    return False


@functools.lru_cache(1)
def getCandidateNoduleList(data_dir):
   """
   Get the list a list of candidate nodules with diameter, center coordinates
   and label (benign / malignant).
   """
   annotations_path = os.path.join(data_dir, 'annotations.csv')
   candidates_path = os.path.join(data_dir, 'candidates.csv')
   nodules = []
   available_uids = getDownloadedSeriesUIDs(data_dir)
   annotations = mapSeriesUIDsToAnnotations(annotations_path)
   print(available_uids)
   with open(candidates_path, 'r') as f:
       for row in list(csv.reader(f))[1:]:
           series_uid = row[0]
           if series_uid not in available_uids:
               continue
           center = tuple([float(x) for x in row[1:4]])
           malignant = bool(row[4])
           for annot in annotations.get(series_uid, []):
               annot_center, annot_diameter = annot
               if not too_far_apart(center, annot_center, annot_diameter/2):
                   nodules.append(
                           CandidateNoduleTuple(
                               malignant,
                               annot_diameter,
                               series_uid,
                               center
                            ))
   return nodules
   

            
            


           

           
    
    
