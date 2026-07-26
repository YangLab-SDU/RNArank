import numpy as np
from pyrosetta import init, pose_from_pdb


init(extra_options="-constant_seed -mute all -read_only_ATOM_entries")


def get_distance_map(pdb_path):
    pose = pose_from_pdb(str(pdb_path))
    coordinates = []
    for index in range(1, pose.size() + 1):
        residue = pose.residue(index)
        coordinates.append(np.asarray(residue.xyz("C4'"), dtype=float))

    coordinates = np.asarray(coordinates)
    differences = coordinates[:, None, :] - coordinates[None, :, :]

    return np.linalg.norm(differences, axis=-1)


def pdb2lddt(native_path, decoy_path, cutoff=30.0):
    native_distances = get_distance_map(native_path)
    decoy_distances = get_distance_map(decoy_path)

    mask = native_distances < cutoff
    np.fill_diagonal(mask, False)
    distance_errors = np.abs(decoy_distances - native_distances)
    thresholds = np.array([1.0, 2.0, 4.0, 6.0])

    pair_scores = (distance_errors[None, :, :] < thresholds[:, None, None]).mean(axis=0)

    contact_counts = mask.sum(axis=0)
    valid_residues = contact_counts > 0

    local_lddt = ((pair_scores * mask).sum(axis=0)[valid_residues]/ contact_counts[valid_residues])

    return local_lddt

if __name__ == "__main__":
    native_path = "7EDL.pdb"
    decoy_path = "7EDL_decoy1.pdb"
    local_lddt = pdb2lddt(native_path, decoy_path)
    print(local_lddt)
    print(np.mean(local_lddt))

