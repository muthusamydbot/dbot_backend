import pandas as pd
import open3d as o3d
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

def ransac_to_dim(filePath):
    width=0.1
    
    DATANAME = filePath
    pcd = o3d.io.read_point_cloud(DATANAME)
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    k_points=7
    pcd = pcd.uniform_down_sample(every_k_points=k_points)


    pcd=np.asarray(pcd.points)
    min_values = np.zeros_like(pcd)
    shifted_point = pcd + min_values - np.amin(pcd)  # Ensure all coordinates become positive
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(shifted_point)


    max_plane_idx = 6
    pt_to_plane_dist = 0.08
    segment_models = {} 
    segments = {}
    rest =  transformed_pcd  
    #axis_point={}
    #bb_box_points=[]

    for i in range (max_plane_idx): 
        colors = plt.get_cmap("tab20")(i) 
        segment_models[i], inliers =rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=1000)
        segments[i]=rest.select_by_index(inliers) 
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)
        segments[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        oriented_bbox = segments[i].get_minimal_oriented_bounding_box()
        #oriented_bbox.color=(0,0,0)
        #bb_box_points.append(np.asarray(oriented_bbox.get_box_points()))
        #axis_point[i]=oriented_bbox
        
        print("pass",i,"/",max_plane_idx, "done.")

    point_11=(segments[0].get_minimal_oriented_bounding_box().center)
    point_12=(segments[2].get_minimal_oriented_bounding_box().center)
    l1 = np.linalg.norm(point_11- point_12)

    point_1=(segments[3].get_minimal_oriented_bounding_box().center)
    point_2=(segments[1].get_minimal_oriented_bounding_box().center)
    height = np.linalg.norm(point_1- point_2)

    point_21=(segments[4].get_minimal_oriented_bounding_box().center)
    point_22=(segments[5].get_minimal_oriented_bounding_box().center)
    l2 = np.linalg.norm(point_21- point_22)


    #print(distance_1,distance_1_2,distance_2_2)

    """
    height=distance_1
    l1=distance_1_2
    l2=distance_2_2
    """

    print(height,l1,l2)

    return height,l1,l2,width




def converter(l1,l2=None):
    w1_p1=(0,0,0)    
    if l2 is None:
        w1_p2=(l1,0,0)
        w2_p1=(l1,0,0)
        w2_p2=(l1,l1,0)
        w3_p1=(l1,l1,0)
        w3_p2=(0,l1,0)
        w4_p1=(0,l1,0)
        w4_p2=(0,0,0)
        floor=(0,l1,0)


    else:

        w1_p2=(l1,0,0)
        w2_p1=(l1,0,0)
        w2_p2=(l1,l2,0)
        w3_p1=(l1,l2,0)
        w3_p2=(0,l2,0)
        w4_p1=(0,l2,0)
        w4_p2=(0,0,0)
        floor=(0,l2,0)

    p1=(w1_p1,w2_p1,w3_p1,w4_p1,w1_p1)
    p2=(w1_p2,w2_p2,w3_p2,w4_p2,floor)
    return p1,p2


def lh_to_excel(l1,width,height,l2=None,folderName='op'):
    p1,p2=converter(l1)
    df=pd.DataFrame()
    df["Object name"]=["Wall 1","Wall 2","Wall 3","Wall 4","Floor"]
    df["Object class"]=("Surface","Surface","Surface","Surface","Surface")
    df["Object type"]=("NA","NA","NA","NA","NA")
    df["Base surface"]=("Floor","Floor","Floor","Floor","NA")
    df["P1"]=p1
    df["P2"]=p2
    df["Width or thickness or depth"]=(width,width,width,width,l1)
    df["Height"]=(height,height,height,height,width)
    output_folder = "./uploaded_files"
    output_file_path = os.path.join(output_folder,folderName,"output.xlsx")
    
    # Save the DataFrame to Excel
    df.to_excel(output_file_path, index=False)

    return output_file_path








class Surface:
    def __init__(self, wall_data):
        self.wall_data = wall_data

    def create_mesh(self, thickness, color):
        
        point1 = ast.literal_eval(self.wall_data.iloc[0, self.wall_data.columns.get_loc('P1')])
        point2 = ast.literal_eval(self.wall_data.iloc[0, self.wall_data.columns.get_loc('P2')])
        height = self.wall_data.iloc[0, self.wall_data.columns.get_loc('Height')]
    
        
        vertices = [
            point1, 
            [point1[0] + thickness, point1[1], point1[2]],
            [point1[0], point1[1], point1[2] + height],
            [point1[0] + thickness, point1[1], point1[2] + height],
            point2,  
            [point2[0] + thickness, point2[1], point2[2]],
            [point2[0], point2[1], point2[2] + height],
            [point2[0] + thickness, point2[1], point2[2] + height],
        ]
    
        triangles = [
            [0, 2, 4], [2, 4, 6],  
            [1, 3, 5], [3, 5, 7],  
            [0, 1, 2], [1, 2, 3],
            [1, 2, 3], [2, 1, 0],
            [2, 1, 0], [3, 2, 1],
            [4, 5, 6], [5, 6, 7],
            [5, 6, 7], [6, 5, 4],
            [6, 5, 4], [7, 6, 5],
            [0, 1, 4], [1, 4, 5],
            [1, 4, 5], [4, 1, 0],
            [4, 1, 0], [5, 4, 1],
            [2, 3, 6], [3, 6, 7], 
            [3, 6, 7], [6, 3, 2],
            [6, 3, 2], [7, 6, 3],
            [2, 6, 4], [3, 7, 5],
            [0, 4, 2], [1, 5, 3],
            [1, 5, 3], [3, 7, 5],
        ]
    
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
        mesh.paint_uniform_color(color)
    
        return mesh


class SurfaceElement:
    def __init__(self, element_data):
        self.element_data = element_data

    def create_mesh(self, color):
        point1 = ast.literal_eval(self.element_data.iloc[0, self.element_data.columns.get_loc('P1')])
        point2 = ast.literal_eval(self.element_data.iloc[0, self.element_data.columns.get_loc('P2')])
        thickness = self.element_data.iloc[0, self.element_data.columns.get_loc('Width or thickness or depth')]
        height = self.element_data.iloc[0, self.element_data.columns.get_loc('Height')]

        vertices = [
            point1,
            [point1[0] + thickness, point1[1], point1[2]],
            [point1[0], point1[1], point1[2] + height],
            [point1[0] + thickness, point1[1], point1[2] + height],
            point2,
            [point2[0] + thickness, point2[1], point2[2]],
            [point2[0], point2[1], point2[2] + height],
            [point2[0] + thickness, point2[1], point2[2] + height]
        ]

        triangles = [
            [0, 2, 4], [2, 4, 6],  
            [1, 3, 5], [3, 5, 7],  
            [0, 1, 2], [1, 2, 3],
            [1, 2, 3], [2, 1, 0],   #
            [2, 1, 0], [3, 2, 1],  #
            [4, 5, 6], [5, 6, 7],
            [5, 6, 7], [6, 5, 4],  #
            [6, 5, 4], [7, 6, 5],   #
            [0, 1, 4], [1, 4, 5],
            [1, 4, 5], [4, 1, 0],     #
            [4, 1, 0], [5, 4, 1],    #
            [2, 3, 6], [3, 6, 7], 
            [3, 6, 7], [6, 3, 2],  #
            [6, 3, 2], [7, 6, 3],  #
            [2, 6, 4], [3, 7, 5],
            [0, 4, 2], [1, 5, 3],
            [1, 5, 3], [3, 7, 5]
        ]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        vertex_colors = np.tile(color, (len(vertices), 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return mesh


class RoomDefiningObject:
    def __init__(self, object_data):
        self.object_data = object_data

    def create_mesh(self, color):
        point1 = ast.literal_eval(self.object_data.iloc[0, self.object_data.columns.get_loc('P1')])
        point2 = ast.literal_eval(self.object_data.iloc[0, self.object_data.columns.get_loc('P2')])
        thickness = self.object_data.iloc[0, self.object_data.columns.get_loc('Width or thickness or depth')]
        height = self.object_data.iloc[0, self.object_data.columns.get_loc('Height')]

        vertices = [
            point1,
            [point1[0] + thickness, point1[1], point1[2]],
            [point1[0], point1[1], point1[2] + height],
            [point1[0] + thickness, point1[1], point1[2] + height],
            point2,
            [point2[0] + thickness, point2[1], point2[2]],
            [point2[0], point2[1], point2[2] + height],
            [point2[0] + thickness, point2[1], point2[2] + height]
        ]

        triangles = [
            [0, 2, 4], [2, 4, 6],  
            [1, 3, 5], [3, 5, 7],  
            [0, 1, 2], [1, 2, 3],
            [1, 2, 3], [2, 1, 0],   #
            [2, 1, 0], [3, 2, 1],  #
            [4, 5, 6], [5, 6, 7],
            [5, 6, 7], [6, 5, 4],  #
            [6, 5, 4], [7, 6, 5],   #
            [0, 1, 4], [1, 4, 5],
            [1, 4, 5], [4, 1, 0],     #
            [4, 1, 0], [5, 4, 1],    #
            [2, 3, 6], [3, 6, 7], 
            [3, 6, 7], [6, 3, 2],  #
            [6, 3, 2], [7, 6, 3],  #
            [2, 6, 4], [3, 7, 5],
            [0, 4, 2], [1, 5, 3],
            [1, 5, 3], [3, 7, 5]
        ]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        vertex_colors = np.tile(color, (len(vertices), 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return mesh


def main(xlPath,folderName= 'op', base_url = ''):
    excel_file = xlPath
    sheet_name = 'Sheet1'
    wall_data = pd.read_excel(excel_file, sheet_name)

    object_name_column = 'Object name'
    thickness_column = 'Width or thickness or depth'

    surfaces = []
    surface_elements = []
    room_defining_objects = []

    
    floor_data_i = wall_data[wall_data[object_name_column] == 'Floor']
    floor_surface = Surface(floor_data_i)
    floor_color = [0.5, 0.5, 0.5]
    floor_mesh = floor_surface.create_mesh(thickness=floor_data_i.iloc[0, floor_data_i.columns.get_loc(thickness_column)], color=floor_color)
    surfaces.append(floor_mesh)

    
    for i in range(1,5):
        wall_name = f'Wall {i}'
        wall_data_i = wall_data[wall_data[object_name_column] == wall_name]
        wall_surface = Surface(wall_data_i)
        wall_color = [0.9, 0.85, 0.8]
        wall_mesh = wall_surface.create_mesh(thickness=wall_data_i.iloc[0, wall_data_i.columns.get_loc(thickness_column)], color=wall_color)
        surfaces.append(wall_mesh)

    #o3d.visualization.draw_geometries(surfaces + surface_elements + room_defining_objects)

    # Export the combined mesh to an STL file
    output_folder = "./uploaded_files"
    output_file_path = os.path.join(output_folder, folderName,"output.stl")
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # Concatenate individual meshes
    for mesh in surfaces + surface_elements + room_defining_objects:
        combined_mesh += mesh

    # Compute normals for the combined mesh
    combined_mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_file_path, combined_mesh)

    print(f"Exported the 3D model to: {output_file_path}")
    file_url = f"{base_url}/uploaded_files/{folderName}/output.stl"
    return file_url


def firstmain(url, folderName, base_url):
    # Set the headers to bypass ngrok browser warning
    headers = {'ngrok-skip-browser-warning': 'true'}

    # Make a request to the provided URL with the custom headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful\
    print("HEllo")
    if response.status_code == 200:
        print("hai")
        # Convert the response content to bytes and create a file
        content = response.content
        with open('temporary_file.ply', 'wb') as f:
            f.write(content)

        # Call your existing function with the downloaded file
        height, l1, l2, width = ransac_to_dim(filePath='temporary_file.ply')
        xlpath = lh_to_excel(l1, width, height, l2, folderName=folderName)
        return main(xlPath=xlpath, folderName=folderName, base_url=base_url)
    else:
        print("Failed to retrieve the file.")


# def firstmain(url,folderName, base_url):
#     height,l1,l2,width=ransac_to_dim(filePath=url)
#     xlpath = lh_to_excel(l1,width,height,l2,folderName=folderName)
#     print(xlpath)
#     return main(xlPath=xlpath,folderName=folderName, base_url=base_url)

