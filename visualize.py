
import os
import numpy as np
import itertools
import math, random
random.seed = 42
import numpy as np
import open3d as o3d

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

import pygem
print(pygem.__version__)
from pygem import FFD
from pygem import CustomDeformation


from path import Path
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import kaleido



def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers',marker=dict(size=8))]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    fig.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False),
                                   yaxis = dict(showgrid = False,showticklabels = False),
                                   zaxis = dict(showgrid = False,showticklabels = False)
        )
        
        )
    fig.show()




def pcwrite(name,xs,ys,zs):
        data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                    mode='markers')]
        fig = visualize_rotate(data)
        fig.update_traces(marker=dict(size=2,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
        fig.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False),
                                   yaxis = dict(showgrid = False,showticklabels = False),
                                   zaxis = dict(showgrid = False,showticklabels = False)
        )
        
        )
        fig.write_image(name+'.jpg')





def pc_show_multi(object_coords,control_point_coords):
    """
    xs,ys,zs the coordinates from ML40 object;
    xp,yp,zp  the coordinated from FFD control points 
    """
    xs,ys,zs = object_coords
    xp,yp,zp = control_point_coords


    
    data = [go.Scatter3d(x=xs, y=ys, z=zs,mode='markers'),go.Scatter3d(x=xp, y=yp, z=zp,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


# def pc_show_multi_list(object_coords_list):
#     """
#     xs,ys,zs the coordinates from ML40 object;
#     xp,yp,zp  the coordinated from FFD control points 
#     """
#     xs,ys,zs = object_coords
#     xp,yp,zp = control_point_coords
#     for object in object_coords_list:
        

    
#     data = [go.Scatter3d(x=xs, y=ys, z=zs,mode='markers'),
#             go.Scatter3d(x=xp, y=yp, z=zp, mode='markers'),
#             go.Scatter3d(x=xp, y=yp, z=zp, mode='markers')]
#     fig = visualize_rotate(data)
#     fig.update_traces(marker=dict(size=2,
#                       line=dict(width=2,
#                       color='DarkSlateGrey')),
#                       selector=dict(mode='markers'))
#     fig.show()



def pc_show_multi_1(object_coords,control_point_coords,p_color):
    """
    xs,ys,zs the coordinates from ML40 object;
    xp,yp,zp  the coordinated from FFD control points 
    """
    xs,ys,zs = object_coords
    xp,yp,zp = control_point_coords


    
    data = [go.Scatter3d(x=xs, y=ys, z=zs,mode='markers'),go.Scatter3d(x=xp, y=yp, z=zp,
                                   mode='markers',marker=dict(color=p_color))]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()




def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    
    fig = go.Figure(data=data,
        # layout=go.Layout(
        #     updatemenus=[dict(type='buttons',
        #         showactive=False,
        #         y=1,
        #         x=0.8,
        #         xanchor='left',
        #         yanchor='bottom',
        #         pad=dict(t=45, r=10),
        #         buttons=[dict(label='Play',
        #             method='animate',
        #             args=[None, dict(frame=dict(duration=50, redraw=True),
        #                 transition=dict(duration=0),
        #                 fromcurrent=True,
        #                 mode='immediate'
        #                 )]
        #             )
        #         ])]
        # ),
        # frames=frames
    )
    fig.update_traces(marker=dict(size=2,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    fig.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False),
                                   yaxis = dict(showgrid = False,showticklabels = False),
                                   zaxis = dict(showgrid = False,showticklabels = False)
        )
        
        )

    return fig






