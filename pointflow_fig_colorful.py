import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

pcd_path = sys.argv[1]
threshold = sys.argv[2]
radius = float(sys.argv[3])

print(radius)

# pcd_path = 'Deform_custom_img/bench/pointcloud/deform_1.npy'
def standardize_bbox(pcl):
    # pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    # np.random.shuffle(pt_indices)
    # pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""
xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
def generate_colormap(data,threshold=0.4,light=None):
    # Define the threshold for special coloring
    threshold = threshold
    n_colors = 2048
    # Define colors: dark blue for below threshold, and orange to dark red for above threshold
    dark_blue = np.array([0, 0, 0.5])   # Dark blue
    light_blue = np.array([0.678,0.847,0.902])
    orange = np.array([1, 0.5, 0]) # Orange
    dark_red = np.array([0.5, 0, 0])  
    # dark_red = np.array([0.2, 0, 0])   # Dark red
    data = data.T
    print(dark_red)
    x, y, z, distance = data[:,0], data[:,1], data[:,2], data[:,3]

    if distance.min()==distance.max():
        cool_colors = np.zeros((n_colors, 3))
        for i in range(3):
           cool_colors[:, i] = np.linspace(dark_blue[i], dark_blue[i], n_colors)
        custom_cmap = ListedColormap(cool_colors)
        return custom_cmap
    else:
        # Calculate the proportion of the colormap dedicated to each part
        proportion = threshold / (np.max(data) - np.min(data))

        # Calculate the number of colors for each part of the colormap
        n_dark_blue = int(n_colors * proportion)
        n_gradient = n_colors - n_dark_blue

        # Create the color arrays
        dark_blue_colors = np.tile(dark_blue, (n_dark_blue, 1))
        gradient_cool_colors = np.zeros((n_dark_blue, 3))
        gradient_warm_colors = np.zeros((n_gradient, 3))
        for i in range(3):
            gradient_warm_colors[:, i] = np.linspace(light_blue[i], dark_red[i], n_gradient)
        for i in range(3):
            gradient_cool_colors[:, i] = np.linspace(dark_blue[i], light_blue[i], n_dark_blue)

        # Combine the color arrays to form the final colormap
        combined_colors = np.vstack((gradient_cool_colors, gradient_warm_colors))
        custom_cmap = ListedColormap(combined_colors)
        return custom_cmap


# def colormap(x,y,z):
#     vec = np.array([x,y,z])
#     vec = np.clip(vec, 0.001,1.0)
#     norm = np.sqrt(np.sum(vec**2))
#     vec /= norm
#     return [vec[0], vec[1], vec[2]]

def colormap(distance, idx ,cmap):
    # Example mapping based on z-coordinate
    # Normalize z to the range [0, 1]
    if distance.min() == distance.max():
         return cmap.colors[0]
    n_colors = len(cmap.colors)
    distance_normalized = (distance - distance.min()) / (distance.max() - distance.min())
    distance_normalized = np.clip(distance_normalized, 0, 1)  
    # Map this normalized value to a color in the colormap
    color_idx = int(distance_normalized[idx] * (n_colors - 1))
    return cmap.colors[color_idx]





xml_segments = [xml_head]

pcl = np.load(pcd_path)
# cp  = np.load('Deform_custom_img/airplane/pointcloud/cp.npy')
# cp  = cp.T
pcl = standardize_bbox(pcl)
custom_cmap = generate_colormap(pcl,float(threshold))
pcl[:,1] *= -1
pcl = pcl * 1.5
# cp[:,1] *= -1

for i in range(pcl.shape[0]):
    # R,G,B
    color = colormap(distance = pcl[:,3], idx =i,cmap=custom_cmap)
    # color = custom_cmap.colors[i]
    xml_segments.append(xml_ball_segment.format(radius,pcl[i,0],pcl[i,1],pcl[i,2], *color))
# for i in range(cp.shape[0]):
#     color = np.array([1, 0.5, 0])
#     xml_segments.append(xml_ball_segment.format(radius * 3,cp[i,0],cp[i,1],cp[i,2], *color))

xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)
classname = pcd_path.split('/')[-3]

deform_type = pcd_path.split('/')[-1].split('.')[0]

folder_name= 'Deform_custom_img'
cls_folder_name=classname
if not os.path.exists(folder_name + '/' + cls_folder_name +'/render_xml' ):
            # If it doesn't exist, create the folder
            os.mkdir(folder_name + '/' + cls_folder_name +'/render_xml' )

print(classname)
with open('Deform_custom_img/{}/render_xml/{}.xml'.format(classname,deform_type), 'w') as f:
    f.write(xml_content)





def generate_mxl(root,pcd,class_name,deform_type,cmap,radius=0.02):
    n_colors=cmap.colors
    xml_head = \
        """
        <scene version="0.5.0">
            <integrator type="path">
                <integer name="maxDepth" value="-1"/>
            </integrator>
            <sensor type="perspective">
                <float name="farClip" value="100"/>
                <float name="nearClip" value="0.1"/>
                <transform name="toWorld">
                    <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
                </transform>
                <float name="fov" value="25"/>
                
                <sampler type="ldsampler">
                    <integer name="sampleCount" value="256"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="1600"/>
                    <integer name="height" value="1200"/>
                    <rfilter type="gaussian"/>
                    <boolean name="banner" value="false"/>
                </film>
            </sensor>
            
            <bsdf type="roughplastic" id="surfaceMaterial">
                <string name="distribution" value="ggx"/>
                <float name="alpha" value="0.05"/>
                <float name="intIOR" value="1.46"/>
                <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
            </bsdf>
            
        """
    xml_ball_segment = \
        """
            <shape type="sphere">
                <float name="radius" value="{}"/>
                <transform name="toWorld">
                    <translate x="{}" y="{}" z="{}"/>
                </transform>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="{},{},{}"/>
                </bsdf>
            </shape>
        """

    xml_tail = \
        """
            <shape type="rectangle">
                <ref name="bsdf" id="surfaceMaterial"/>
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <translate x="0" y="0" z="-0.5"/>
                </transform>
            </shape>
            
            <shape type="rectangle">
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
                </transform>
                <emitter type="area">
                    <rgb name="radiance" value="6,6,6"/>
                </emitter>
            </shape>
        </scene>
        """

    xml_segments = [xml_head]
    try:
        os.path.exists(os.path.join(root,class_name))
    except:
        print('Create the file for point cloud first')

    pcl = pcd
    pcl = standardize_bbox(pcl)
    cmap = generate_colormap(pcl)
    pcl[:,1] *= -1

    for i in range(pcl.shape[0]):
        # R,G,B
        color = colormap(distance = pcl[:,3], idx =i,cmap=cmap)
        # color = custom_cmap.colors[i]
        xml_segments.append(xml_ball_segment.format(radius,pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)
    classname = class_name
    print(classname)
    with open(root+"/"+class_name+"/"+'render_xml/{}.xml'.format(deform_type), 'w') as f:
        f.write(xml_content)