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

def extract_clash(pdict):
    radius = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8, }
    pose = pdict['pose']
    nres = pdict['nres']
    atom_name_list = []
    atom_xyz_list = []
    bonding_list = []
    residue_idx_list = []
    idx = 0
    for i in range(1, nres + 1):
        residue_idx_list.append([i, idx])
        r = pose.residue(i)
        O3_idx = None
        P_idx = None
        for j in range(1, r.natoms() + 1):
            atom_name = r.atom_name(j).strip()
            if atom_name[0] in ["C", "N", "O", "P"]:
                idx += 1
                atom_name_list.append([i, atom_name, atom_name[0], radius[atom_name[0]]])
                atom_xyz_list.append(np.array(r.atom(j).xyz()))
                if atom_name == "O3\'":
                    O3_idx = idx
                if atom_name == "P":
                    P_idx = idx
        bonding_list.append([O3_idx, P_idx])
    residue_idx_list = np.array(residue_idx_list)
    atom_xyz_list = np.array(atom_xyz_list)
    atom_name_list = np.array(atom_name_list)
    bonding_list = np.array(bonding_list)

    dis = distance.cdist(atom_xyz_list, atom_xyz_list, 'euclidean')
    dis_threshold = np.zeros_like(dis)
    for i in range(dis_threshold.shape[0]):
        for j in range(dis_threshold.shape[0]):
            dis_threshold[i, j] = float(atom_name_list[i, 3]) + float(atom_name_list[j, 3]) - 0.4
    clash_1hot = dis < dis_threshold
    np.fill_diagonal(clash_1hot, False)
    for i in range(bonding_list.shape[0] - 1):
        clash_1hot[bonding_list[i, 0], bonding_list[i + 1, 1]] = False
    for i in range(residue_idx_list.shape[0]):
        start_idx = residue_idx_list[i, 1]
        if i < residue_idx_list.shape[0] - 1:
            end_idx = residue_idx_list[i + 1, 1]
        else:
            end_idx = clash_1hot.shape[0]
        clash_1hot[start_idx:end_idx, start_idx:end_idx] = False
    clash_num = np.zeros((residue_idx_list.shape[0], residue_idx_list.shape[0]))
    atom_num = np.zeros((residue_idx_list.shape[0], residue_idx_list.shape[0]))
    for i in range(residue_idx_list.shape[0]):
        start_idx_i = residue_idx_list[i, 1]
        if i < residue_idx_list.shape[0] - 1:
            end_idx_i = residue_idx_list[i + 1, 1]
        else:
            end_idx_i = clash_1hot.shape[0]
        for j in range(residue_idx_list.shape[0]):
            start_idx_j = residue_idx_list[j, 1]
            if j < residue_idx_list.shape[0] - 1:
                end_idx_j = residue_idx_list[j + 1, 1]
            else:
                end_idx_j = clash_1hot.shape[0]
            atom_num[i, j] = clash_1hot[start_idx_i:end_idx_i, start_idx_j:end_idx_j].size
            clash_num[i, j] = np.sum(clash_1hot[start_idx_i:end_idx_i, start_idx_j:end_idx_j])
    clash_prob = clash_num / atom_num
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

def get_f2d(msa):
    nrow, ncol = msa.shape[-2:]
    if nrow == 1:
        msa = np.repeat(msa.reshape(nrow, ncol), 2, axis=0)
        nrow = 2
    msa1hot = (np.arange(5) == msa[..., None]).astype(float)  # (h, L, 5)
    w = reweight(msa1hot, .8)
    f1d_seq = msa1hot[0, :, :4]  # (L, 4)
    f1d_pssm = msa2pssm(msa1hot, w)  # (L, 6)
    f1d = np.concatenate([f1d_seq, f1d_pssm], axis=1)
    f2d_dca = fast_dca(msa1hot, w)  # (L, L, 26)
    f2d = np.concatenate([f1d[:, None, :].repeat(ncol, axis=1), f1d[None, :, :].repeat(ncol, axis=0), f2d_dca], axis=-1)

    return f1d, f2d

def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = np.tensordot(msa1hot, msa1hot, axes=([1, 2], [1, 2]))
    id_mask = id_mtx > id_min
    w = 1.0 / id_mask.sum(axis=-1).astype(float)

    return w

def msa2pssm(msa1hot, w):
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(axis=0) / beff + 1e-9
    h_i = (-f_i * np.log(f_i)).sum(axis=1)

    return np.concatenate([f_i, h_i[:, None]], axis=1)

def fast_dca(msa1hot, weights, penalty=4.5):
    nr, nc, ns = msa1hot.shape
    x = msa1hot.reshape(nr, nc * ns)
    num_points = weights.sum() - np.sqrt(weights.mean())
    mean = (x * weights[:, None]).sum(axis=0, keepdims=True) / num_points
    x = (x - mean) * np.sqrt(weights[:, None])
    cov = np.matmul(x.T, x) / num_points
    cov_reg = cov + np.eye(nc * ns) * penalty / np.sqrt(weights.sum())
    inv_cov = np.linalg.inv(cov_reg)
    x1 = inv_cov.reshape(nc, ns, nc, ns)
    x2 = x1.transpose(0, 2, 1, 3)
    features = x2.reshape(nc, nc, ns * ns)
    x3 = np.sqrt((x1[:, :-1, :, :-1] ** 2).sum(axis=(1, 3))) * (1 - np.eye(nc))
    apc = x3.sum(axis=0, keepdims=True) * x3.sum(axis=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - np.eye(nc))

    return np.concatenate([features, contacts[:, :, None]], axis=2)

def pose2fd(pdict):
    mapping = {'RAD': 0, 'URA': 1, 'RCY': 2, 'RGU': 3}
    nucles = []
    length = int(pdict['pose'].total_residue())
    for i in range(length):
        index1 = i + 1
        nt_name = pdict['pose'].residue(index1).name().split(":")[0].split("_")[0]
        nucles.append(mapping[nt_name])
    nucles = np.array(nucles, dtype=np.uint8)[None, :]
    f1d_nucles, f2d_nucles = get_f2d(nucles)
    return f1d_nucles, f2d_nucles

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
