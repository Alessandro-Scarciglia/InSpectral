import numpy as np
import open3d as o3d

# Parameters
RAY = 10.

# Funzione per generare i punti sulla sfera
def generate_points_on_sphere(N):
    phi = np.random.uniform(0, 2 * np.pi, N)  # Angolo azimutale
    cos_theta = np.random.uniform(-1, 1, N)   # Coseno dell'angolo zenitale
    theta = np.arccos(cos_theta)              # Angolo zenitale

    # Coordinate cartesiane
    x = RAY * np.sin(theta) * np.cos(phi)
    y = RAY * np.sin(theta) * np.sin(phi)
    z = RAY * np.cos(theta)

    return np.vstack((x, y, z)).T  # Restituisce un array Nx3

# Funzione per ottenere la trasformazione punto-to-mondo
def point_to_world_transform(x, y, z):
    
    normal = np.array([x, y, z])  # Direzione dell'asse Z
    normal = normal / np.linalg.norm(normal)  # Normalizza il vettore normale

    # Vettore arbitrario per trovare assi ortogonali
    arbitrary_vector = np.array([1.0, 0.0, 0.0])
    if np.allclose(normal, arbitrary_vector):
        arbitrary_vector = np.array([1.0, 0.0, 0.0])
    
    # Calcola il vettore X ortogonale alla normale
    x_axis = np.cross(arbitrary_vector, normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Calcola il vettore Y ortogonale sia alla normale che a X
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Matr. di rotazione (R)
    R = np.vstack([x_axis, y_axis, normal]).T  # 3x3 matrice di rotazione

    # Traslazione
    T = np.array([x, y, z])

    # Crea la matrice di trasformazione omogenea 4x4
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R  # Ruota
    transform_matrix[:3, 3] = T   # Traslazione
    
    return transform_matrix

# Funzione per salvare la sequenza di trasformazioni
def save_transformations_to_npy(filename, transformations):
    # Appiattiamo ogni matrice 4x4 in una sequenza di 16 valori
    flattened_transformations = [matrix.reshape(-1) for matrix in transformations]
    # Converte la lista in un array numpy e la salva su file
    np.save(filename, np.array(flattened_transformations))

# Funzione per visualizzare la sfera con i frame
def visualize_points_and_frames(points, transformations):
    # Crea una palla (sfera)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RAY)
    sphere.compute_vertex_normals()
    
    # Crea un punto cloud per i punti sulla sfera
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    
    # Crea i frame da visualizzare
    frames = []
    for transform in transformations:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)  # Assi X, Y, Z
        frame.transform(transform)  # Applica la trasformazione al frame
        frames.append(frame)
    
    # Visualizza la sfera, i punti e i frame
    o3d.visualization.draw_geometries([sphere, pc] + frames)

# Esempio: generare 1000 punti sulla sfera
N = 5
points = generate_points_on_sphere(N)

# Calcolare la trasformazione per ogni punto
transformations = []
for point in points:
    transform = point_to_world_transform(*point)
    transformations.append(transform)

# Visualizza i punti e i frame
visualize_points_and_frames(points, transformations)

# Salva le trasformazioni come sequenze da 16 valori
filename = '/home/visione/Projects/InSpectral/data/test_wps.npy'
save_transformations_to_npy(filename, transformations)
print(f"Trasformazioni salvate in {filename}")
