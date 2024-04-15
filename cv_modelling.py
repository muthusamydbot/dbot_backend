import torch
import os
import numpy as np
from mask3d import get_model, load_mesh, prepare_data #, map_output_to_pointcloud, save_colorized_mesh 
import open3d as o3d
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mask3d.datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_20, 
    SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_200, 
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200)

########################################################################################################################
def input_process(input_file):
   
    model = get_model('scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load input data
    mesh = load_mesh(input_file)
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    #return data, points, colors, features, unique_map, inverse_map
    return mesh,outputs,inverse_map,points

########################################################################################################################

# map output to point cloud
def map_output_to_pointcloud(mesh, 
                             outputs, 
                             inverse_map,points,
                             label_space='scannet200',
                             confidence_threshold=0.9):
    
    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map])  # mapping the mask back to the original point cloud
    

    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    # save labelled mesh    
    labels_mapped = np.zeros((len(mesh.vertices), 1))
    instance_mapped = np.zeros((len(mesh.vertices), 1))

    colors_mapped = np.zeros((len(mesh.vertices), 3))
    col_inst_mapped=np.zeros((len(mesh.vertices), 2))
    scannet_class_lab=np.zeros((len(mesh.vertices), 1))
    item_counts = {}  # Dictionary to store the count of each item
    result = []       # List to store the result
    
    for i, (l, c, m) in enumerate(sorted(zip(labels, confidences, masks_binary), key=lambda x: x[1], reverse=False)):
        if label_space == 'scannet200':
            label_offset = 2
            if l == 0:
                l = -1 + label_offset
            else:
                l = int(l) + label_offset
            
            if l not in item_counts:  # First instance of the item
                item_counts[l] = 1
                result.append((l, 1))
                val=1
            else:  
                count = item_counts[l] + 1
                item_counts[l] = count
                result.append((l, count))
                val=count

        labels_mapped[m == 1] = l
        colors_mapped[m == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]]

        instance_mapped[m==1] = val
        col_inst_mapped[m==1]=[l,val]
        v_li = CLASS_LABELS_200[int(l)]
        scannet_class_lab[m==1]=VALID_CLASS_IDS_200[l]
        
        print(v_li,val,l,"############",VALID_CLASS_IDS_200[l])
    
    seg_point = [[point, np.array(col, dtype=int)] for point, col in zip(points, col_inst_mapped)]
    return col_inst_mapped,seg_point,labels_mapped,instance_mapped,scannet_class_lab

########################################################################################################################

def seg_info(labels_mapped,points):

    def filter_point_cloud(ck, threshold=-1.1):
        indices = np.where(ck[:, 2] >= threshold)
        filtered_ck = ck[indices]
        only_wall_ply = o3d.geometry.PointCloud()
        only_wall_ply.points = o3d.utility.Vector3dVector(filtered_ck)
    
        return only_wall_ply

    wall = []
    door = []
    for i, instance in enumerate(labels_mapped):
        if instance == 0:
            wall.append(points[i])
        if instance == 4:
            door.append(points[i])

    walls_array=np.array(wall)
    wall_ply = o3d.geometry.PointCloud()
    wall_ply.points = o3d.utility.Vector3dVector(walls_array)
    wall_ply, ind = wall_ply.remove_statistical_outlier(nb_neighbors=200,
                                                        std_ratio=2.0)

    door_array=np.array(door)
    door_ply_imp = o3d.geometry.PointCloud()
    door_ply_imp.points = o3d.utility.Vector3dVector(door)
    door_ply, ind = door_ply_imp.remove_statistical_outlier(nb_neighbors=200,
                                                        std_ratio=2.0)

    door_array=np.asarray(door_ply.points)
    door_bbox = door_ply.get_minimal_oriented_bounding_box()
    door_bbox.color=(255,0,255)
  
    only_wall_array = filter_point_cloud(np.asarray(wall_ply.points))
    return only_wall_array,door_array,door_bbox



########################################################################################################################
def ransac(only_wall):
    max_plane_idx = 8
    pt_to_plane_dist = 0.05

    segment_models = {} 
    segments = {}
    rest = only_wall
    bb_box_points=[]
    segments_points={}

    axis_point={}
    for i in range (max_plane_idx): 
        colors = plt.get_cmap("tab20")(i) 
        segment_models[i], inliers =rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
        segments[i]=rest.select_by_index(inliers) 
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True) 
        #oriented_bbox.color=(0,0,0)
        print("pass",i,"/",max_plane_idx, "done.")
        
        oriented_bbox = segments[i].get_minimal_oriented_bounding_box()
        oriented_bbox.color=(0,0,0)
        bb_box_points.append(np.asarray(oriented_bbox.get_box_points()))
        axis_point[i]=oriented_bbox
        segments_points[i]=np.asarray(segments[i].points)

    return segments
    

########################################################################################################################





# o3d.visualization.draw_geometries([seg[i] for i in range(8)])


def main(url,folderName,base_url):
    output_folder = "./uploaded_files"
    output_file_path = os.path.join(output_folder, folderName,"output.ply")
    mesh,outputs,inverse_map,points=input_process(url)
    col_inst_mapped,seg_point,labels_mapped,instance_mapped,scannet_class_lab= map_output_to_pointcloud(mesh,
                                                                                                        outputs,
                                                                                                        inverse_map,
                                                                                                        points)
    only_wall,door_array,door_bbox=seg_info(labels_mapped,points)
    seg=ransac(only_wall)


    o3d.io.write_point_cloud(output_file_path,seg)
    file_url = f"{base_url}/uploaded_files/{folderName}/output.ply"
    return file_url

# Estimapcdte normals
#filtered_ck.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Visualize the point cloud with normals


