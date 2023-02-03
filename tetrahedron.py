import jax
from jax import random, jit, vmap
import jax.numpy as np

@jit
def orthonormal_vector(p1, p2):
    v2 = np.cross(p1, p2)

    v2_norm = np.linalg.norm(v2)
    v2 = v2 / v2_norm

    return v2

@jit
def calculate_angle(v1, v2):
    # v1, v2: [N, 3]
    # return: [N]
    return np.arccos(np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))

@jit
def calc_basis(normal, orthogonal_vector):
    # Normalize the normal
    normal = normal / np.linalg.norm(normal)
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    # Calculate the orthogonal vector to the plane
    other_orthogonal = np.cross(normal, orthogonal_vector)
    
    # Normalize the orthogonal vector
    other_orthogonal = other_orthogonal / np.linalg.norm(other_orthogonal)
    
    # Return the basis vectors as columns in a matrix
    return np.column_stack((other_orthogonal, orthogonal_vector))

@jit
def rotate_2d(v, rad):
    # v: [N, 2]
    # rad: [N]
    # return: [N, 2]
    cos = np.cos(rad)
    sin = np.sin(rad)
    return np.stack([v[..., 0] * cos - v[..., 1] * sin, v[..., 0] * sin + v[..., 1] * cos], axis=-1)

@jit
def calculate_angle(v1, v2):
# v1, v2: [N, 3]
# return: [N]
    return np.arccos(np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))

@jit
def compute_tetrahedron(b):
    # split into p1 and p2
    v1 = b[:3]
    p = b[3:]

    v1_norm = np.linalg.norm(v1)
    v2 = orthonormal_vector(v1, p) * v1_norm * ((2 * np.sqrt(2)/3))

    basis = calc_basis(v1, v2)
    inv_basis = np.linalg.pinv(basis)
    
    v2_2d = inv_basis @ v2

    v3 = rotate_2d(v2_2d, 2*np.pi/3)
    v4 = rotate_2d(v2_2d, 4*np.pi/3)

    v3 = basis @ v3
    v4 = basis @ v4

    v3 = v3 / np.linalg.norm(v3) * np.linalg.norm(v2)
    v4 = v4 / np.linalg.norm(v4) * np.linalg.norm(v2)

    v2 = v2 - v1 / 3
    v3 = v3 - v1 / 3
    v4 = v4 - v1 / 3

    return np.stack([v1, v2, v3, v4], axis=0)

if __name__ == "__main__":
    key = random.PRNGKey(0)
    rand_key, subkey = random.split(key)
    # generate a random vector of norm np.sqrt(3/2)
    v1 = random.normal(key, [2, 3])
    v1 = v1 / np.expand_dims(np.linalg.norm(v1[:, :3], axis = 1), axis=1) *  5
    p = random.normal(subkey, [2, 3])
    b = np.concatenate([v1, p], axis = 1)
    # b = np.array([[1, 0, 0, 3, 0, 0]])

    compute_tetrahedron_batched = vmap(compute_tetrahedron, in_axes=0, out_axes=1)
    v1, v2, v3, v4 = compute_tetrahedron_batched(b)


    print("v1:", v1)
    print("v2:", v2)
    print("v3:", v3)
    print("v4:", v4)

    # norm of v2
    print("norm of v1:", np.linalg.norm(v1, axis = 1))
    print("norm of v2:", np.linalg.norm(v2, axis = 1))
    print("norm of v3:", np.linalg.norm(v3, axis = 1))
    print("norm of v4:", np.linalg.norm(v4, axis = 1))

    print("angle v1 v2:", calculate_angle(v1, v2))
    print("angle v1 v3:", calculate_angle(v2, v3))
    print("angle v1 v4:", calculate_angle(v2, v4))
