"""
Taken from https://github.com/szymanowiczs/viewset-diffusion
"""

from pytorch3d.vis import plotly_vis
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.camera_utils import join_cameras_as_batch

import torch

def normalize_sequence(dataset, sequence_name, volume_side_length, vis=False):
    """
    Normalizes the sequence using the point cloud information. Takes 3 steps to normalize
    the cameras so that the point clouds are aligned across sequences.
    1. Normalize translation: shift cameras and point cloud so that COM is at origin
    2. Normalize rotation: using photographer's bias, align the point cloud with y-axis
    3. Normalize scale so that point cloud fits in a cube of side length volume_side_length
    """
    needs_checking = False
    frame_idx_gen = dataset.sequence_indices_in_order(sequence_name)
    frame_idxs = []
    while True:
        try:
            frame_idx = next(frame_idx_gen)
            frame_idxs.append(frame_idx)
        except StopIteration:
            break

    cameras_start = []
    for frame_idx in frame_idxs:
        cameras_start.append(dataset[frame_idx].camera)
    cameras_start = join_cameras_as_batch(cameras_start)
    cameras = cameras_start.clone()

    # ===== Translation normalization
    point_cloud_pts = dataset[frame_idxs[0]].sequence_point_cloud.points_list()[0].clone()
    # find the center of mass
    com = torch.mean(point_cloud_pts, dim=0)
    # center the point cloud
    point_cloud_pts = point_cloud_pts - com
    # shift the cameras accordingly
    cameras.T = torch.matmul(com, cameras.R) + cameras.T
    
    # ===== Rotation normalization
    # Estimate the world 'up' direction assuming that yaw is small
    # and running SVD on the x-vectors of the cameras
    x_vectors = cameras.R.transpose(1, 2)[:, 0, :].clone()
    x_vectors -= x_vectors.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x_vectors)
    V = Vh.mH
    # vector with the smallest variation is to the normal to
    # the plane of x-vectors (assume this to be the up direction)
    if S[0] / S[1] > S[1] / S[2]:
        print('Warning: unexpected singular values in sequence {}: {}'.format(sequence_name, S))
        needs_checking = True
    estimated_world_up = V[:, 2:]
    # check all cameras have the same y-direction
    for camera_idx in range(len(cameras.T)):
        if torch.sign(torch.dot(estimated_world_up[:, 0],
                                cameras.R[0].transpose(0,1)[1, :])) != torch.sign(torch.dot(estimated_world_up[:, 0],
                                    cameras.R[camera_idx].transpose(0,1)[1, :])):
            print("Some cameras appear to be flipped in sequence {}".format(sequence_name) )
            needs_checking = True
    flip = torch.sign(torch.dot(estimated_world_up[:, 0], cameras.R[0].transpose(0,1)[1, :])) < 0
    if flip:
        estimated_world_up = V[:, 2:] * -1
    # build the target coordinate basis using the estimated world up
    target_coordinate_basis = torch.cat([V[:, :1],
                                        estimated_world_up,
                                        torch.linalg.cross(V[:, :1], estimated_world_up, dim=0)],
                                        dim=1)
    cameras.R = torch.matmul(target_coordinate_basis.T, cameras.R)
    point_cloud_pts = torch.bmm(point_cloud_pts.unsqueeze(1),
                                target_coordinate_basis.unsqueeze(0).expand(len(point_cloud_pts),
                                                                            3, 3)).squeeze(1)
    
    # ===== Scale normalization
    # align the center along the longest axis to the origin
    ranges = torch.max(point_cloud_pts, dim=0)[0] - torch.min(point_cloud_pts, dim=0)[0]
    max_range_index = 1 # torch.argmax(ranges)
    aligned_com_dist = torch.max(point_cloud_pts, dim=0)[0][max_range_index] - ranges[max_range_index] / 2
    aligned_com = torch.zeros(3)
    aligned_com[max_range_index] = aligned_com_dist
    # shift cameras and point cloud
    cameras.T = torch.matmul(aligned_com, cameras.R) + cameras.T
    point_cloud_pts = point_cloud_pts - aligned_com

    max_point_cloud = torch.max(torch.abs(point_cloud_pts))

    scaling_factor = volume_side_length * 0.95 / (2 * max_point_cloud )
    point_cloud_pts = point_cloud_pts * scaling_factor
    cameras.T = cameras.T * scaling_factor

    normalized_point_cloud = Pointclouds([point_cloud_pts])
    maximum_distance = torch.max(torch.norm(cameras.T, dim=1))
    minimum_distance = torch.min(torch.norm(cameras.T, dim=1))

    if vis:
        x_axis_points = torch.tensor([[x, 0, 0] for x in torch.linspace(0, 10, 100)])
        y_axis_points = torch.tensor([[0, y, 0] for y in torch.linspace(0, 10, 100)])
        z_axis_points = torch.tensor([[0, 0, z] for z in torch.linspace(0, 10, 100)])

        axis_dict = {"x_axis": Pointclouds([x_axis_points]),
                    "y_axis": Pointclouds([y_axis_points]), 
                    "z_axis": Pointclouds([z_axis_points]),
        }

        fig = plotly_vis.plot_scene({
            sequence_name:{
                # "point cloud before": dataset[frame_idxs[0]].sequence_point_cloud,
                "point cloud after": normalized_point_cloud,
                # "cameras before": cameras_start,
                "cameras after": cameras,
                **axis_dict
            }
        },
        axis_args=plotly_vis.AxisArgs(showgrid=True))
        fig.show()

    return cameras, minimum_distance, maximum_distance, normalized_point_cloud, needs_checking