import torch
import os
import mdtraj as md
import numpy as np
import math
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from transformers import VivitImageProcessor, VivitModel
import h5py
from .utils import get_gpu_usage_smi, get_memory_usage_gb
import argparse


class ProteinMDDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        self.original_samples = samples
    def __len__(self):
           return len(self.original_samples)
    def __getitem__(self, idx):
        pid, seq, rep = self.original_samples[idx]
        return pid, seq, rep


def custom_collate(batch):
    pid, seq, rep = zip(*batch)
    # Remove the extra dimension from each contact sample
    # traj_list = [traj.squeeze(0) for traj in reps]
    # Now each traj has shape [32, 3, 224, 224]
    # Stack them along a new first dimension (batch dimension)
    # traj_batch = torch.stack(traj_list, dim=0)
    batched_data = {'pid': pid, 'seq': seq, 'traj': rep}
    # return pid, seq, contacts
    return batched_data

def _split_generator(configs):
    """A local generator for random_split, independent of the global RNG state.

    The global RNG state at the point random_split() runs depends on everything that
    happened before it (model init, including how many parameters the Houlsby adapters
    randomly initialize — which varies with num_end_adapter_layers across configs). So even
    with the same fix_seed, two configs with different architectures can end up with
    different train/val/test splits. Using a dedicated generator seeded from a fixed,
    config-independent value makes the split reproducible across runs/configs.
    """
    seed = getattr(configs.train_settings, 'split_seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def prepare_replicate(configs, train_repli_path, test_repli_path):
    samples = prepare_samples(train_repli_path)
    total_samples = len(samples)
    val_size = int(total_samples * 0.05)
    test_size = int(total_samples * 0.05)
    train_size = total_samples - val_size - test_size
    train_samples, val_samples, test_samples = random_split(
        samples, [train_size, val_size, test_size], generator=_split_generator(configs))

    samples_hard = prepare_samples(test_repli_path)
    hard_num = len(samples_hard)
    val_size = int(hard_num * 0.5)
    test_size = hard_num - val_size
    val_hard, test_hard = random_split(
        samples_hard, [val_size, test_size], generator=_split_generator(configs))

    val_samples = val_samples + val_hard
    test_samples = test_samples + test_hard
    print(f"train samples: {len(train_samples)}, val samples: {len(val_samples)}, test samples: {len(test_samples)}")
    # Create DataLoader for each split
    train_dataset = ProteinMDDataset(train_samples, configs=configs, mode="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=False
    )
    val_dataset = ProteinMDDataset(val_samples, configs=configs, mode="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs.valid_settings.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=custom_collate,
        drop_last=False
    )
    test_dataset = ProteinMDDataset(test_samples, configs=configs, mode="val")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=custom_collate,
        drop_last=False
    )
    # dataset = ProteinMDDataset(samples, configs=configs, mode = "train")
    # dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader

def prepare_dataloaders(configs):
    train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0 = prepare_replicate(configs,
                                                                          configs.train_settings.Atlas_data_repli_0_path,
                                                                          configs.train_settings.Atlas_test_repli_0_path)
    train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1 = prepare_replicate(configs,
                                                                          configs.train_settings.Atlas_data_repli_1_path,
                                                                          configs.train_settings.Atlas_test_repli_1_path)
    train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2 = prepare_replicate(configs,
                                                                          configs.train_settings.Atlas_data_repli_2_path,
                                                                          configs.train_settings.Atlas_test_repli_2_path)
    return ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
            (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
            (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))


def prepare_dataloaders_mdcath(configs):
    """Like prepare_dataloaders, but for mdCATH's N replicate paths (a YAML list under
    train_settings.mdCATH_data_repli_path / mdCATH_test_repli_path, not N separate scalar
    keys like Atlas). Returns a list of N (train_dl, val_dl, test_dl) tuples instead of a
    fixed 3-tuple."""
    train_paths = configs.train_settings.mdCATH_data_repli_path
    test_paths = configs.train_settings.mdCATH_test_repli_path
    assert len(train_paths) == len(test_paths), \
        "mdCATH_data_repli_path and mdCATH_test_repli_path must have the same length"
    return [prepare_replicate(configs, train_paths[i], test_paths[i])
            for i in range(len(train_paths))]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-positive training: all R replicates per protein returned together
# ─────────────────────────────────────────────────────────────────────────────

class AtlasMultiReplicateDataset(Dataset):
    """Each item returns (pid, seq, [traj_r0, traj_r1, …]) loaded on demand."""

    def __init__(self, samples):
        # samples: list of (pid, seq, h5_path_r0, h5_path_r1, …)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pid, seq = item[0], item[1]
        paths = item[2:]
        trajs = []
        for path in paths:
            with h5py.File(path, 'r') as f:
                trajs.append(f[pid]['pooled'][:])   # numpy (768,)
        return pid, seq, trajs


def _find_h5_for_pid(pid, dirpath):
    """Return the absolute path of the H5 file for `pid` in `dirpath`, or None."""
    for fname in os.listdir(dirpath):
        if fname.endswith('.h5') and '#'.join(fname.split('#')[:2]) == pid:
            return os.path.abspath(os.path.join(dirpath, fname))
    return None


def collate_multipos(batch):
    """Collate for AtlasMultiReplicateDataset.

    Returns {'pid', 'seq', 'traj'} where 'traj' is a list of R numpy arrays
    each of shape (B, 768) — one per replicate.
    """
    pids = [b[0] for b in batch]
    seqs = [b[1] for b in batch]
    R = len(batch[0][2])
    trajs = [np.stack([b[2][r] for b in batch], axis=0) for r in range(R)]
    return {'pid': pids, 'seq': seqs, 'traj': trajs}


def _build_multipos_samples(data_paths, hard_paths=None):
    """Find proteins present in ALL replicate directories and return sample tuples.

    Each sample: (pid, seq, h5_path_r0, h5_path_r1, h5_path_r2).
    Proteins missing from any replicate are silently skipped.
    """
    R = len(data_paths)
    samples = []
    n_skipped = 0

    ref_dir = data_paths[0]
    for fname in sorted(os.listdir(ref_dir)):
        if not fname.endswith('.h5'):
            continue
        pid = '#'.join(fname.split('#')[:2])
        path_r0 = os.path.abspath(os.path.join(ref_dir, fname))

        other_paths = []
        for r in range(1, R):
            p = _find_h5_for_pid(pid, data_paths[r])
            if p is None:
                break
            other_paths.append(p)

        if len(other_paths) < R - 1:
            n_skipped += 1
            continue

        # Read seq from replicate 0 (identical across replicates)
        with h5py.File(path_r0, 'r') as f:
            seq = f[pid].attrs['seq']

        samples.append((pid, seq, path_r0, *other_paths))

    print(f"[multi_positive] {len(samples)} proteins with all {R} replicates "
          f"({n_skipped} skipped — missing ≥1 replicate)")

    if hard_paths is not None:
        hard_samples = []
        n_hard_skipped = 0
        ref_hard = hard_paths[0]
        for fname in sorted(os.listdir(ref_hard)):
            if not fname.endswith('.h5'):
                continue
            pid = '#'.join(fname.split('#')[:2])
            path_r0 = os.path.abspath(os.path.join(ref_hard, fname))
            other_paths = []
            for r in range(1, R):
                p = _find_h5_for_pid(pid, hard_paths[r])
                if p is None:
                    break
                other_paths.append(p)
            if len(other_paths) < R - 1:
                n_hard_skipped += 1
                continue
            with h5py.File(path_r0, 'r') as f:
                seq = f[pid].attrs['seq']
            hard_samples.append((pid, seq, path_r0, *other_paths))
        print(f"[multi_positive] hard set: {len(hard_samples)} proteins "
              f"({n_hard_skipped} skipped)")
        return samples, hard_samples

    return samples, []


def prepare_dataloaders_multipos(configs):
    """Return (train_loader, val_loader, test_loader) with all 3 replicates per item.

    Replaces the 3 sequential `prepare_replicate` calls.  The batch 'traj' field
    is a list of R numpy arrays each of shape (B, 768) instead of a single tuple.
    Backward-compatible: only used when train_settings.multi_positive: True.
    """
    train_paths = [
        configs.train_settings.Atlas_data_repli_0_path,
        configs.train_settings.Atlas_data_repli_1_path,
        configs.train_settings.Atlas_data_repli_2_path,
    ]
    hard_paths = [
        configs.train_settings.Atlas_test_repli_0_path,
        configs.train_settings.Atlas_test_repli_1_path,
        configs.train_settings.Atlas_test_repli_2_path,
    ]

    all_samples, hard_samples = _build_multipos_samples(train_paths, hard_paths)

    total = len(all_samples)
    val_size  = int(total * 0.05)
    test_size = int(total * 0.05)
    train_size = total - val_size - test_size
    train_s, val_s, test_s = random_split(
        all_samples, [train_size, val_size, test_size], generator=_split_generator(configs))

    if hard_samples:
        n_hard = len(hard_samples)
        val_hard_size  = n_hard // 2
        test_hard_size = n_hard - val_hard_size
        val_hard, test_hard = random_split(
            hard_samples, [val_hard_size, test_hard_size], generator=_split_generator(configs))
        val_s  = list(val_s)  + list(val_hard)
        test_s = list(test_s) + list(test_hard)

    print(f"[multi_positive] train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    bsz = configs.train_settings.batch_size
    nw  = getattr(configs.train_settings, 'num_workers', 1)

    train_loader = DataLoader(
        AtlasMultiReplicateDataset(list(train_s)),
        batch_size=bsz, shuffle=True,
        collate_fn=collate_multipos, drop_last=False, num_workers=nw,
    )
    val_loader = DataLoader(
        AtlasMultiReplicateDataset(list(val_s)),
        batch_size=bsz, shuffle=False,
        collate_fn=collate_multipos, drop_last=False, num_workers=nw,
    )
    test_loader = DataLoader(
        AtlasMultiReplicateDataset(list(test_s)),
        batch_size=bsz, shuffle=False,
        collate_fn=collate_multipos, drop_last=False, num_workers=nw,
    )
    return train_loader, val_loader, test_loader


def vectors_to_contact_maps(vecs):
    """
    Converts multiple vectors of pairwise contact values into symmetric contact map matrices.
    
    Args:
        vecs (np.ndarray): 2D array of shape (n, L),
                           where L = N*(N-1)/2 and n is number of vectors.
    
    Returns:
        contact_maps (np.ndarray): 3D array of shape (n, N, N).
    """
    n, num_pairs = vecs.shape
    # 计算N
    N = int((1 + math.sqrt(1 + 8 * num_pairs)) / 2)
    
    # 初始化输出数组
    contact_maps = np.zeros((n, N, N), dtype=vecs.dtype)
    
    # 获取上三角索引（不包括对角线）
    triu_indices = np.triu_indices(N, k=1)
    
    # 给所有矩阵的上三角赋值
    contact_maps[:, triu_indices[0], triu_indices[1]] = vecs
    
    # 镜像到下三角，使矩阵对称
    contact_maps = contact_maps + np.transpose(contact_maps, (0, 2, 1))
    
    return contact_maps
    

# def pdb2seq(pdb_path):
#     # Create a PDB parser object
#     parser = PDBParser(QUIET=True)
    
#     # Parse the structure from the PDB file
#     structure = parser.get_structure("protein", pdb_path)
    
#     # Initialize the polypeptide builder
#     ppb = PPBuilder()
    
#     # Extract sequences from all chains in the structure
#     for model in structure:
#         for chain in model:
#             # Build polypeptides for the chain (could be more than one segment)
#             polypeptides = ppb.build_peptides(chain)
#             for poly_index, poly in enumerate(polypeptides):
#                 sequence = poly.get_sequence()
#                 # print(f"Chain {chain.id} (segment {poly_index}): {sequence}")
#     return sequence
def pdb2seq(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    
    sequences = []
    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            seq = "".join(str(poly.get_sequence()) for poly in polypeptides)
            sequences.append(seq)
    
    return "".join(sequences)

def normalize_contact_map(contact_map):
    # Option 1: Divide by maximum value (if maximum is known and consistent)
    max_val = contact_map.max()  # or a fixed value like 6.28 if appropriate
    normalized_map = contact_map / max_val
    # Optionally clip to [0,1] to ensure all values are within range
    normalized_map = np.clip(normalized_map, 0, 1)
    return normalized_map

def prepare_samples(datapath):
    samples=[]
    # Sorted so the index ordering random_split() operates over is stable/reproducible —
    # os.listdir() order is not guaranteed across machines/filesystem states.
    for file_name in sorted(os.listdir(datapath)):
        file_path = os.path.abspath(os.path.join(datapath, file_name))
        pid, seq, rep = loadh5(file_path)
        samples.append((pid, seq, rep))

    return samples


def process_samples(data_folder2search, model_name, sub_list, output_folder):
    image_processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitModel.from_pretrained(model_name)
    # Specify the parent folder (folder A)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VivitModel.from_pretrained(model_name).to(device)
    folder_A = data_folder2search
    samples = []
    
    # Iterate through each subfolder under folder_A
    #data={}
    for subfolder in os.listdir(folder_A):
        subfolder_path = os.path.abspath(os.path.join(folder_A, subfolder))
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path) and subfolder_path in sub_list:
            traj_files=[]
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".xtc"):
                    file_xtc_path = os.path.join(subfolder_path, file_name)
                    traj_files.append(file_xtc_path)
                    # num_xtc += 1
                elif file_name.endswith(".pdb"):
                    file_pdb_path = os.path.join(subfolder_path, file_name)
                    pid = file_name.split(".")[0]
                    # num_pdb += 1
            print(pid)
            sequence = pdb2seq(file_pdb_path)
            #fragments, frag_contact = fragment_sequence(sequence, np.array([]), fixed_length=224, overlap=30)
            #data_len=len(fragments)
            rep=0
            
            for file_xtc_path in traj_files:
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]

                traj = md.load(file_xtc_path, top=file_pdb_path)
                if len(traj)<32:
                    continue
                # cutoff = 7  # in nanometers
                # contact_maps = []
                #traj_stride=1000
                #traj_num=traj.n_frames//traj_stride
                #traj_remain=traj.n_frames%traj_stride
                #traj_list=[]
                #for i in range(traj_num):
                #    distances = md.compute_distances(traj[i*traj_stride:i*traj_stride+traj_stride], 
                #    traj[i*traj_stride:i*traj_stride+traj_stride].topology.select_pairs('name CA', 'name CA'))
                #    contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #    traj_list.append(contact_maps)
                #
                #if traj_remain!=0:
                #    distances = md.compute_distances(traj[-traj_remain:], traj[-traj_remain:].topology.select_pairs('name CA', 'name CA'))
                #    contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #    traj_list.append(contact_maps)
                #
                #contact_maps = np.concatenate(traj_list, axis=0)
                
                #distances = md.compute_distances(traj, traj.topology.select_pairs('name CA', 'name CA'))
                #contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #print(contact_maps.dtype)
                #print(contact_maps.shape)
                #fragments, frag_contacts = fragment_sequence(sequence, contact_maps, fixed_length=224, overlap=30)
                fragments, frag_contacts = fragment_sequence(sequence, traj, fixed_length=224, overlap=30)
                #print(frag_contacts[0].dtype)
                for i in range(len(fragments)):
                    frag_seq=fragments[i]
                    frag_contact=frag_contacts[i]
                    frag_contact = np.expand_dims(frag_contact, axis=-1) # [n, N, N, 1]
                    frag_contact = normalize_contact_map(frag_contact) # [n, N, N, 1]
                    frag_contact = np.repeat(frag_contact, 3, axis=-1) # [n, N, N, 3]
                    #data[pid]['ind2seq'][i]=frag_seq
                    #data[pid]['ind2map'][i]=frag_contact
                #for i in range(len(fragments)):
                    pid_i = pid+"#frag"+str(i)
                    h5id = pid_i+"#rep"+str(rep)
                    #seq_i = data[pid]['ind2seq'][i]
                    seq_i = frag_seq
                    # traj = image_processor(data[pid]['ind2map'][i], return_tensors="pt", do_rescale=False, offset=False, 
                    #                        do_normalize=False, do_resize=True, size=[224, 224]) # [1, 1000, 3, 224, 224]
                    images = image_processor(list(frag_contact), return_tensors="pt", do_rescale=False, offset=False, 
                                           do_normalize=False, do_resize=False) # [1, 1000, 3, 224, 224]
                    images = images['pixel_values']
                    #print(images.dtype)
                    groups = group_frames(images)
                    #print(groups.dtype)
                    groups = groups.to(device)
                    sub_num=groups.shape[0]//8
                    sub_remain=groups.shape[0]%8
                    rep_list=[]
                    print(get_memory_usage_gb())
                    for z in range(sub_num):
                        #print(z)
                        sub_groups = groups[z*8:z*8+8]
                        with torch.no_grad():
                            vivit_output = model(sub_groups)
                        last_hidden_states = vivit_output.last_hidden_state #[list_len, 3137, 768]
                        cls_representation = last_hidden_states[:, 0, :] #[list_len, 1, 768]
                        # sub_list.append(cls_representation)
                        rep_list.append(cls_representation.detach().cpu())
                        #print(get_memory_usage_gb())
                    
                    #cls_tem=torch.cat(rep_list, dim=0) #[list_len, 1, 768]
                    if sub_remain!=0:
                        sub_groups = groups[-sub_remain:]
                        with torch.no_grad():
                            vivit_output = model(sub_groups)
                        last_hidden_states = vivit_output.last_hidden_state #[list_len, 3137, 768]
                        cls_representation = last_hidden_states[:, 0, :] #[list_len, 1, 768]
                        rep_list.append(cls_representation.detach().cpu())
                        
                    
                    #cls_representation = torch.stack(rep_list, axis=0) #[list_len,  768]
                    cls_representation = torch.cat(rep_list, dim=0) #[list_len, 768]
                    #print(cls_representation.shape)
                    pooled = cls_representation.mean(axis=0)  # shape: (768,)
                    #print(pooled.shape)
                    # samples.append((h5id, pid_i, seq_i, pooled))
                    saveh5((pid_i, seq_i, pooled), os.path.join(output_folder, h5id+".h5"))
                    del rep_list, images, groups, frag_contact
                    torch.cuda.empty_cache()
                    print(get_memory_usage_gb())
                rep+=1
                del traj
                torch.cuda.empty_cache()
                print(get_memory_usage_gb())
    

def fragment_sequence(sequence, traj, fixed_length, overlap):
    fragments = []
    frag_contact = []
    stride = fixed_length - overlap
    start = 0
    seq_len = len(sequence)
    while start < seq_len:
        end = min(start + fixed_length, seq_len)
        fragment = sequence[start:end]
        fragments.append(fragment)  # Add fragment *before* checking length
        atom_indices = traj.topology.select(f"resid {start} to {end-1}")
        sub_traj = traj.atom_slice(atom_indices)
        distances = md.compute_distances(sub_traj, sub_traj.topology.select_pairs('name CA', 'name CA'))
        contact_map = vectors_to_contact_maps(distances) # [n, N, N]
        print(start, end)
        print(contact_map.shape)
        if contact_map.size!=0:
            if end-start<fixed_length:
                background = np.zeros((contact_map.shape[0], 224, 224), dtype=contact_map.dtype)
                #background[:, :end-start, :end-start] = contact_map[:, start:end, start:end]
                background[:, :end-start, :end-start] = contact_map
                frag_contact.append(background)
            else:
                #frag_contact.append(contact_map[:, start:end, start:end])
                frag_contact.append(contact_map)
        if end == seq_len: #check if the end of the sequence has been reached
            break
        start += stride
    return fragments, frag_contact


def saveh5(sample, h5file):
    (pid, seq, pooled) = sample
    # outname = h5file + str()
    os.makedirs(os.path.dirname(h5file), exist_ok=True)
    with h5py.File(h5file, 'w') as f:
    
        # Convert torch tensor to numpy
        grp = f.create_group(pid)
        # grp.attrs['pid'] = pid.encode('utf-8')
        # grp = f.create_group(str(pid))
        grp.create_dataset('pooled', data=pooled, compression='gzip')
        # Save sequence as attribute or dataset (if string, encode first)
        grp.attrs['seq'] = str(seq).encode('utf-8')

def loadh5(h5file):
    with h5py.File(h5file, 'r') as f:
        name = os.path.basename(h5file)
        pid = "#".join(name.split('#')[:2])
        # pid=f[pid].attrs['pid']
        # pid = 'some_pid'
        rep = f[pid]['pooled'][:]
        seq = f[pid].attrs['seq']
    return pid, seq, rep


def group_frames(traj, group_size=32): # traj => [1, 1001, 3, 224, 224]
    """
    Group frames into sets of specified size.
    
    Args:
        total_frames (int): Total number of frames
        group_size (int): Number of frames in each group
        
    Returns:
        list: List of lists, where each inner list is a group of frames
    """
    total_frames = traj.shape[1]
    groups = []
    
    # Calculate how many complete groups we'll have
    num_complete_groups = total_frames // group_size
    
    # Group the frames
    for i in range(num_complete_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = traj[:,start_idx:end_idx,:,:,:]
        groups.append(group)
    
    # Handle any remaining frames
    remaining_frames = total_frames % group_size
    if remaining_frames > 0:
        group = traj[:,total_frames-32:,:,:,:]
        groups.append(group)
    
    processed_arrays = [arr.squeeze(0) for arr in groups]  # shape becomes [32, 3, 224, 224]
    # Stack along a new axis (axis=0)
    result = torch.stack(processed_arrays, axis=0) # [list_len, 32, 3, 224, 224]
    return result

def split_subfolders_into_n_lists(folder_path, n=6):
    # 获取folder_path下所有一级子文件夹（完整路径）
    all_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, d))]
    # 按字母排序（可选）
    all_subfolders.sort()
    # 创建n个空列表
    lists = [[] for _ in range(n)]
    # 均匀分配
    for i, subfolder in enumerate(all_subfolders):
        lists[i % n].append(os.path.abspath(subfolder))
    
    return lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--data_folder2search", help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/')
    parser.add_argument("--model_name", default='google/vivit-b-16x2-kinetics400', help='xx')
    parser.add_argument("--list_num", type=int, default=0, help="xx")
    parser.add_argument("--output_folder", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data', help="xx")
    parser.add_argument("--split_num", type=int, default=6, help="xx")

    
    args_main = parser.parse_args()
    lists = split_subfolders_into_n_lists(args_main.data_folder2search, n=args_main.split_num)
    process_samples(args_main.data_folder2search, args_main.model_name,
                     sub_list=lists[args_main.list_num], output_folder=args_main.output_folder)
