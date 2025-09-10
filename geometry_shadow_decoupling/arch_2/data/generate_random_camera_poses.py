'''
Set of functions to create random point on a Sphere surface.
It is done to define the set of poses for making free inference.
'''

import numpy as np


def generate_points_on_sphere(
        ray: float,
        num_points: int
) -> np.ndarray:
    """
    Function to generate #num_points distributed randomly on a sphere centred in 0 
    and with radiu 'ray'.

    Parameters:
    ----------
    ray: float
        it is the radius of the sphere (unit id up to the user)
    num_points: int
        it is the number of points to be generated

    Returns:
    -------
    stacked_coords: np.ndarray[float]
        it is a structure containing all the (x, y, z) coordinates generated stacked vertically
    """

    # Azimuth
    phi = np.random.uniform(0, 2 * np.pi, num_points)  

    # Zenit
    cos_theta = np.random.uniform(-1, 1, num_points)   
    theta = np.arccos(cos_theta)             

    # Cartesian Coordinates
    x = ray * np.sin(theta) * np.cos(phi)
    y = ray * np.sin(theta) * np.sin(phi)
    z = ray * np.cos(theta)

    # Stack all coordinates
    stacked_coords = np.vstack((x, y, z)).T

    return stacked_coords

def point_to_world_transform(
        x: float,
        y: float,
        z: float
) -> np.ndarray:
    """
    This function assigns a reference frame to a point, so that the z-axis points to the origin.

    Parameters:
    ----------
    x: float
        coordinate on x-axis
    y: float
        coordinate on y-axis
    z: float
        coordinate on z-axis

    Returns:
    -------
    transform_matrix: np.ndarray[float]
        the homogeneous 4x4 matrix representing the point frame in world frame
    """
    
    # Z-axis direction
    normal = np.array([x, y, z])
    normal = normal / np.linalg.norm(normal)

    # Arbitrary vector to find the other axis
    arbitrary_vector = np.array([1.0, 0.0, 0.0])
    if np.allclose(normal, arbitrary_vector):
        arbitrary_vector = np.array([1.0, 0.0, 0.0])
    
    # Compute x-axis
    x_axis = np.cross(arbitrary_vector, normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Compute y-axis (for right-hand frames)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Generate rotation matrix
    R = np.vstack([x_axis, y_axis, normal]).T

    # Generate translation vector
    T = np.array([x, y, z])

    # Build the related homogeneous matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R  
    transform_matrix[:3, 3] = T  
    
    return transform_matrix

def save_transformations_to_npy(
        filename: str,
        transformations
 ) -> None:
    """
    This function store the transformation matrices of waypoints in a .npy file.

    Parameters:
    ----------
    filename: str
        the output filename along with the root path
    transformations: np.ndarray[np.ndarray[float]]
        the set of all generated waypoints
    """

    # Flattening the matrices from 4x4 to 16x1
    flattened_transformations = [matrix.reshape(-1) for matrix in transformations]

    # Convert in numpy array and store
    np.save(filename, np.array(flattened_transformations))


if __name__ == "__main__":

    # Set the number of camera poses
    num_points = 5
    ray = 10
    do_store_wp = False
    dst_filename = '/home/visione/Projects/InSpectral/data/test_wps.npy'

    # Generate points
    points = generate_points_on_sphere(ray=ray, num_points=num_points)

    # Compute transformation for each point
    transformations = []
    for point in points:
        transform = point_to_world_transform(*point)
        transformations.append(transform)

    # Store waypoints generated
    if do_store_wp:
        save_transformations_to_npy(dst_filename, transformations)
        print(f"Transformations stored at {dst_filename}")
