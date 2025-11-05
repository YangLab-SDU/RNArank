import argparse
import numpy as np
from scipy.spatial import distance, distance_matrix
import time
from Bio.PDB.PDBParser import PDBParser
from PixelateResidue import *

from pyrosetta import *
init(extra_options="-constant_seed -mute all -read_only_ATOM_entries")
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose.rna import *
from pyrosetta.rosetta.core.pose import *

def rbf(D):
    num_rbf = 16
    D_min, D_max, D_count = 0., 100., num_rbf
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape((1, 1, -1))
    D_sigma = (D_max - D_min) / D_count
    D_expand = np.expand_dims(D, axis=-1)
    RBF = np.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_distmaps(pose, atom1="P", atom2="P", default="C4'"):
    psize = pose.size()
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(1, psize + 1):
        r = pose.residue(i)
        if type(atom1) == str:
            if r.has(atom1):
                xyz1[i - 1, :] = np.array(r.xyz(atom1))
            else:
                xyz1[i - 1, :] = np.array(r.xyz(default))
        else:
            xyz1[i - 1, :] = np.array(r.xyz(atom1.get(r.name(), default)))
        if type(atom2) == str:
            if r.has(atom2):
                xyz2[i - 1, :] = np.array(r.xyz(atom2))
            else:
                xyz2[i - 1, :] = np.array(r.xyz(default))
        else:
            xyz2[i - 1, :] = np.array(r.xyz(atom2.get(r.name(), default)))
    return distance_matrix(xyz1, xyz2)

def extract_nucs(pdict):
    nucles = []
    nuclt = ['RAD', 'RGU', 'RCY', 'URA']
    residuemap = dict([(nuclt[i], i) for i in range(len(nuclt))])
    length = int(pdict['pose'].total_residue())
    for i in range(length):
        index1 = i + 1
        nucles.append(pdict['pose'].residue(index1).name().split(":")[0].split("_")[0])
    _prop = np.zeros((5, len(nucles)))
    for i in range(len(nucles)):
        nuc = nucles[i]
        if nuc in residuemap:
            _prop[residuemap[nuc], i] = 1
        _prop[-1, i] = min(i, len(nucles) - i) * 1.0 / len(nucles) * 2
    pdict['seq'] = _prop

def extract_multi_distance_map(pdict):
    x1 = get_distmaps(pdict['pose'], atom1="C4'", atom2="C4'")
    x2 = get_distmaps(pdict['pose'], atom1="N", atom2="N")
    x3 = get_distmaps(pdict['pose'], atom1="P", atom2="P")
    pdict['dismaps'] = np.concatenate((rbf(x1), rbf(x2), rbf(x3)), axis=-1)



def set_features1D(pdict):
    pose = pdict['pose']
    nres = pdict['nres']
    pdict['alpha'] = np.array(np.deg2rad([pose.alpha(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['gamma'] = np.array(np.deg2rad([pose.gamma(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['beta'] = np.array(np.deg2rad([pose.beta(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['delta'] = np.array(np.deg2rad([pose.delta(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['epsilon'] = np.array(np.deg2rad([pose.epsilon(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['zeta'] = np.array(np.deg2rad([pose.zeta(i) for i in range(1, nres + 1)])).astype(np.float32)
    pdict['chi'] = np.array(np.deg2rad([pose.chi(i) for i in range(1, nres + 1)])).astype(np.float32)


def energy_maps(pdict):
    energy_terms = [core.scoring.ScoreType.fa_atr, core.scoring.ScoreType.fa_rep, core.scoring.ScoreType.lk_nonpolar,
                    core.scoring.ScoreType.rna_torsion, core.scoring.ScoreType.fa_stack,
                    core.scoring.ScoreType.stack_elec, core.scoring.ScoreType.geom_sol_fast,
                    core.scoring.ScoreType.hbond_sc, core.scoring.ScoreType.fa_elec_rna_phos_phos]
    sf = core.scoring.ScoreFunctionFactory.create_score_function("stepwise/rna/rna_res_level_energy4.wts")
    sf(pdict['pose'])
    length = int(pdict['pose'].total_residue())

    tensor = np.zeros((len(energy_terms), length, length))
    energy_graph = pdict['pose'].energies().energy_graph()
    for index, energy_term in enumerate(energy_terms):
        for ii in range(1, pdict['pose'].size() + 1):
            for jj in range(1, pdict['pose'].size() + 1):
                edge = energy_graph.find_energy_edge(ii, jj)
                if (edge != None):
                    emap = edge.fill_energy_map()
                    resid1 = str(ii) + " " + pdict['pose'].residue(ii).name1()
                    resid2 = str(jj) + " " + pdict['pose'].residue(jj).name1()
                    resid_pair = resid1 + " " + resid2
                    score = emap[energy_term]
                    tensor[index, ii - 1, jj - 1] = score
    pdict['energy'] = tensor

def extract_clash(pdict, clash_buffer=0.4):
    radius = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8}
    pose = pdict['pose']
    nres = pdict['nres']

    atom_xyz_list = []
    atom_radii = []
    residue_start_idx = []
    bonding_pairs = []

    idx = 0
    for i in range(1, nres + 1):
        residue_start_idx.append(idx)
        r = pose.residue(i)
        O3_idx, P_idx = None, None
        for j in range(1, r.natoms() + 1):
            atom_name = r.atom_name(j).strip()
            if atom_name[0] in radius:
                atom_xyz_list.append(np.array(r.atom(j).xyz()))
                atom_radii.append(radius[atom_name[0]])
                idx += 1
                if atom_name == "O3'":
                    O3_idx = idx - 1
                if atom_name == "P":
                    P_idx = idx - 1
        bonding_pairs.append([O3_idx, P_idx])
    residue_start_idx.append(idx)

    atom_xyz = np.array(atom_xyz_list)
    atom_radii = np.array(atom_radii)
    bonding_pairs = np.array(bonding_pairs, dtype=object)

    dis = distance.cdist(atom_xyz, atom_xyz, 'euclidean')

    dis_threshold = atom_radii[:, None] + atom_radii[None, :] - clash_buffer

    clash_1hot = dis < dis_threshold
    np.fill_diagonal(clash_1hot, False)

    for bp in bonding_pairs:
        if bp[0] is not None and bp[1] is not None:
            clash_1hot[bp[0], bp[1]] = False
            clash_1hot[bp[1], bp[0]] = False

    residue_starts = np.array(residue_start_idx)
    for i in range(nres):
        s, e = residue_starts[i], residue_starts[i + 1]
        clash_1hot[s:e, s:e] = False

    clash_prob = np.zeros((nres, nres))
    for i in range(nres):
        s_i, e_i = residue_starts[i], residue_starts[i + 1]
        for j in range(i, nres):
            s_j, e_j = residue_starts[j], residue_starts[j + 1]
            block = clash_1hot[s_i:e_i, s_j:e_j]
            total = block.size
            num = np.count_nonzero(block)
            p = num / total if total > 0 else 0.0
            clash_prob[i, j] = clash_prob[j, i] = p  # 对称
    pdict['clash_prob'] = clash_prob

def extract_USR(pdict):
    hang = pdict['pose'].size()
    distance = get_distmaps(pdict['pose'], atom1="C4'", atom2="C4'")
    avg1 = []
    avg2 = []
    avg3 = []
    for i in range(hang):
        avg1.append(np.average(distance[i]))
        idx2 = np.argmax(distance, axis=1)
        avg2.append(np.average(distance[idx2[i]]))
        idx3 = np.argmax(distance[idx2[i]], axis=0)
        avg3.append(np.average(distance[idx3]))
    usr = np.concatenate((rbf(avg1), rbf(avg2), rbf(avg3)), axis=-1)
    pdict['usr'] = usr

def get_coords(pdict):
    nres = pdict["pose"].size()
    pdict["C4'"] = np.stack([np.array(pdict["pose"].residue(i).atom("C4\'").xyz()) for i in range(1, nres + 1)])
    pdict["P"] = np.stack([np.array(pdict["pose"].residue(i).atom("P").xyz()) for i in range(1, nres + 1)])
    pdict["N"] = np.zeros((nres, 3))
    for i in range(1, nres + 1):
        r = pdict["pose"].residue(i)
        if r.name3().strip() in ["A", "G"]:
            pdict["N"][i - 1, :] = np.array(r.xyz("N9"))
        elif r.name3().strip() in ["C", "U"]:
            pdict["N"][i - 1, :] = np.array(r.xyz("N1"))

def set_lframe(pdict):
    n1, n2 = pdict["N"] - pdict["P"], pdict["C4'"] - pdict["P"]
    n1 = np.array(n1)
    n2 = np.array(n2)
    e1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True)
    e2 = n2 - np.einsum('...d,...d->...', n2, e1)[..., None] * e1
    e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2, axis=-1)
    e3 = e3 / np.linalg.norm(e3, axis=-1, keepdims=True)
    pdict["frame_P"] = np.stack([e1, e2, e3], axis=-1)

    n1, n2 = pdict["N"] - pdict["C4'"], pdict["P"] - pdict["C4'"]
    n1 = np.array(n1)
    n2 = np.array(n2)
    e1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True)
    e2 = n2 - np.einsum('...d,...d->...', n2, e1)[..., None] * e1
    e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2, axis=-1)
    e3 = e3 / np.linalg.norm(e3, axis=-1, keepdims=True)
    pdict["frame_C4'"] = np.stack([e1, e2, e3], axis=-1)

def voxelization(path, pdict):
    length = pdict['nres']
    ori = pdict["C4'"]
    frame_C4_ = pdict["frame_C4'"]
    parser = PDBParser(QUIET=True)
    model = parser.get_structure('', path)
    pixels = np.zeros((length, 3, NBINS, NBINS, NBINS))
    pixels = pixelate_atoms_in_box(model, pixels, ori, frame_C4_)
    pdict['pixels'] = pixels

def pose2fd(pdict):
    mapping = {'RAD': 0, 'URA': 1, 'RCY': 2, 'RGU': 3}
    nucles = []
    length = int(pdict['pose'].total_residue())
    for i in range(length):
        nt_name = pdict['pose'].residue(i + 1).name().split(":")[0].split("_")[0]
        nucles.append(mapping[nt_name])
    nucles = np.array(nucles, dtype=np.uint8)[None, :]  # (1, L)
    f1d, f2d = get_f2d_single(nucles)
    return f1d, f2d

def get_f2d_single(msa):
    _, ncol = msa.shape
    L = ncol
    ns = 5

    msa1hot_4ch = (np.arange(4) == msa[..., None]).astype(float)[0]  # (L, 4)
    f1d_seq = msa1hot_4ch

    f_i_5ch = np.pad(f1d_seq, ((0, 0), (0, 1)), 'constant')  # (L, 5)
    h_i = np.zeros((L, 1))
    f1d_pssm = np.concatenate([f_i_5ch, h_i], axis=1)  # (L, 6)

    f1d = np.concatenate([f1d_seq, f1d_pssm], axis=1)  # (L, 10)
    penalty = 4.5
    weights = np.array([0.5, 0.5])
    num_points = weights.sum() - np.sqrt(weights.mean())  # 1.0 - sqrt(0.5)

    x_flat = f_i_5ch.reshape(L * ns)
    mean = x_flat[None, :] / num_points

    v = (x_flat - mean) * np.sqrt(weights[0])
    v = v.flatten()

    alpha = 2.0 / num_points
    beta = penalty / np.sqrt(weights.sum())  # beta = penalty

    vTv = np.dot(v, v)
    c1 = 1.0 / beta
    c2 = alpha / (beta ** 2 + beta * alpha * vTv)

    inv_cov_flat = -c2 * np.outer(v, v)
    diag_indices = np.arange(L * ns)
    inv_cov_flat[diag_indices, diag_indices] += c1

    x1 = inv_cov_flat.reshape(L, ns, L, ns)
    x2 = x1.transpose(0, 2, 1, 3)
    features = x2.reshape(L, L, ns * ns)

    x3 = np.sqrt((x1[:, :-1, :, :-1] ** 2).sum(axis=(1, 3))) * (1 - np.eye(L))
    apc = x3.sum(axis=0, keepdims=True) * x3.sum(axis=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - np.eye(L))

    f2d_dca = np.concatenate([features, contacts[:, :, None]], axis=2)  # (L, L, 26)

    f2d = np.concatenate([f1d[:, None, :].repeat(L, axis=1), f1d[None, :, :].repeat(L, axis=0), f2d_dca], axis=-1)

    return f1d, f2d

def ExtractFeature(args):
    output = args.output
    os.makedirs(output, exist_ok=True)
    pdb = args.pdb
    pose = pose_from_pdb(pdb)
    pdict = {}
    pdict['pose'] = pose
    pdict['nres'] = pdict['pose'].size()
    extract_nucs(pdict)
    extract_multi_distance_map(pdict)
    set_features1D(pdict)
    energy_maps(pdict)
    extract_clash(pdict)
    extract_USR(pdict)
    get_coords(pdict)
    set_lframe(pdict)
    voxelization(path=pdb, pdict=pdict)
    f1d, f2d = pose2fd(pdict)
    np.savez_compressed(os.path.join(output, pdb.split("/")[-1].split(".")[0] + ".npz"),
                        seq=pdict['seq'].astype(np.float32),
                        dismaps=pdict['dismaps'].astype(np.float32),
                        alpha=pdict['alpha'].astype(np.float32),
                        gamma=pdict['gamma'].astype(np.float32),
                        beta=pdict['beta'].astype(np.float32),
                        delta=pdict['delta'].astype(np.float32),
                        epsilon=pdict['epsilon'].astype(np.float32),
                        zeta=pdict['zeta'].astype(np.float32),
                        chi=pdict['chi'].astype(np.float32),
                        energy=pdict['energy'].astype(np.float32),
                        clash_prob=pdict['clash_prob'].astype(np.float32),
                        usr=pdict['usr'].astype(np.float32),
                        pixels=pdict['pixels'].astype(np.float32),
                        f1d=f1d.astype(np.float32),
                        f2d=f2d.astype(np.float32))

def getData(pdb):
    pose = pose_from_pdb(pdb)
    pdict = {}
    pdict['pose'] = pose
    pdict['nres'] = pdict['pose'].size()
    extract_nucs(pdict)
    extract_multi_distance_map(pdict)
    set_features1D(pdict)
    energy_maps(pdict)
    extract_clash(pdict)
    extract_USR(pdict)
    get_coords(pdict)
    set_lframe(pdict)
    voxelization(path=pdb, pdict=pdict)
    pixels = pdict['pixels']
    f1d, f2d = pose2fd(pdict)
    angles = np.stack([np.sin(pdict["alpha"]), np.cos(pdict["alpha"]),
                       np.sin(pdict["gamma"]), np.cos(pdict["gamma"]),
                       np.sin(pdict["beta"]), np.cos(pdict["beta"]),
                       np.sin(pdict["delta"]), np.cos(pdict["delta"]),
                       np.sin(pdict["epsilon"]), np.cos(pdict["epsilon"]),
                       np.sin(pdict["zeta"]), np.cos(pdict["zeta"]),
                       np.sin(pdict["chi"]), np.cos(pdict["chi"])], axis=-1)
    usr = pdict['usr']
    seq = pdict['seq'].T
    dismaps = pdict['dismaps']
    clash_prob = np.expand_dims(pdict['clash_prob'], -1)
    energy = pdict['energy'].T

    fea_1d_model1 = np.concatenate((angles, usr[0], f1d), axis=-1)
    fea_2d_model1 = np.concatenate((dismaps, clash_prob, energy, f2d), axis=-1)
    fea_2d_model1 = np.expand_dims(fea_2d_model1.transpose(2, 0, 1), 0)

    fea_1d_model2 = np.concatenate((angles, seq, usr[0]), axis=-1)
    fea_2d_model2 = np.concatenate((dismaps, clash_prob, energy), axis=-1)
    fea_2d_model2 = np.expand_dims(fea_2d_model2.transpose(2, 0, 1), 0)

    return pixels, fea_1d_model1, fea_2d_model1, fea_1d_model2, fea_2d_model2


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser("Feature.py")
    parser.add_argument('--pdb', required=True, help="Path to pdb file")
    parser.add_argument('--output', default='./Feature/', help="Path to save feature file, default='./Feature'")
    args = parser.parse_args()
    ExtractFeature(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTime: {elapsed_time:.5f} seconds\n")
