import torch
from torch_geometric.data import Data
import numpy as np



def obs_to_pyg_data(obs, env, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = int(obs.dim_topo)

    # Build node features (example: p, rho, danger)
    p     = torch.zeros(N)
    rho   = torch.zeros(N)
    danger= torch.zeros(N)

    p[env.action_space.gen_pos_topo_vect]      = torch.tensor(obs.gen_p)
    p[env.action_space.load_pos_topo_vect]     = torch.tensor(obs.load_p)
    p[env.action_space.line_or_pos_topo_vect]  = torch.tensor(obs.p_or)
    p[env.action_space.line_ex_pos_topo_vect]  = torch.tensor(obs.p_ex)

    rho[env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.rho)
    rho[env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.rho)

    dang = (torch.tensor(obs.rho) >= 0.98).float()
    danger[env.action_space.line_or_pos_topo_vect] = dang
    danger[env.action_space.line_ex_pos_topo_vect] = dang

    x = torch.stack([p, rho, danger], dim=1).to(device)  # [N, 3]

    # Build undirected edge_index from connectivity
    A = torch.tensor(obs.connectivity_matrix(), dtype=torch.float32)
    iu = torch.triu(A, diagonal=1).nonzero(as_tuple=False)  # [E, 2]
    if iu.numel() == 0:
        # fallback for isolated nodes
        edge_index = torch.empty(2,0, dtype=torch.long)
    else:
        e1 = iu.t()                          # [2, E]
        e2 = iu.flip(dims=[1]).t()           # [2, E]
        edge_index = torch.cat([e1, e2], dim=1)  # [2, 2E]

    data = Data(x=x, edge_index=edge_index.to(device))
    return data




@torch.no_grad()
def build_homogeneous_grid_graph(
    obs,
    env,
    device=None,
    danger_thresh: float = 0.98,
    add_self_loops: bool = False,
    include_busbar_adj: bool = False,
):
    """
    Build a homogeneous PyG graph from an L2RPN observation.

    Nodes: topology positions (N = obs.dim_topo)
    Edges: transmission lines (both directions). Optionally busbar adjacency.

    Node features (d=11):
        [0] p                 (MW)
        [1] q                 (MVAr)
        [2] v                 (p.u.)     (gen/load/line-end if available)
        [3] theta             (rad)      (gen/load/line-end if available)
        [4] rho               (per-unit line loading mapped to both endpoints)
        [5] line_status       (0/1)      (for line endpoints)
        [6] cooldown_line     (steps)    (for line endpoints)
        [7] danger_flag       (1 if rho>=danger_thresh else 0) at endpoints
        [8] is_gen_node       (0/1)
        [9] is_load_node      (0/1)
        [10] is_line_endpoint (0/1)

    Edge attributes (k=5):
        [0] rho
        [1] p_flow
        [2] q_flow
        [3] status (0/1)
        [4] dir    (+1 for or->ex, -1 for ex->or)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Sizes & mappings ----------
    N = int(obs.dim_topo)

    pos_gen = np.asarray(env.action_space.gen_pos_topo_vect)       # [n_gen]
    pos_load = np.asarray(env.action_space.load_pos_topo_vect)     # [n_load]
    pos_or = np.asarray(env.action_space.line_or_pos_topo_vect)    # [n_line]
    pos_ex = np.asarray(env.action_space.line_ex_pos_topo_vect)    # [n_line]

    n_line = len(pos_or)

    # ---------- Raw obs arrays (numpy -> torch) ----------
    # Generators / Loads
    gen_p     = torch.as_tensor(getattr(obs, "gen_p",   np.zeros(len(pos_gen))), dtype=torch.float32)
    gen_q     = torch.as_tensor(getattr(obs, "gen_q",   np.zeros(len(pos_gen))), dtype=torch.float32)
    gen_v     = torch.as_tensor(getattr(obs, "gen_v",   np.zeros(len(pos_gen))), dtype=torch.float32)
    gen_theta = torch.as_tensor(getattr(obs, "gen_theta",np.zeros(len(pos_gen))), dtype=torch.float32)

    load_p     = torch.as_tensor(getattr(obs, "load_p",   np.zeros(len(pos_load))), dtype=torch.float32)
    load_q     = torch.as_tensor(getattr(obs, "load_q",   np.zeros(len(pos_load))), dtype=torch.float32)
    load_v     = torch.as_tensor(getattr(obs, "load_v",   np.zeros(len(pos_load))), dtype=torch.float32)
    load_theta = torch.as_tensor(getattr(obs, "load_theta",np.zeros(len(pos_load))), dtype=torch.float32)

    # Lines
    p_or     = torch.as_tensor(getattr(obs, "p_or",   np.zeros(n_line)), dtype=torch.float32)
    q_or     = torch.as_tensor(getattr(obs, "q_or",   np.zeros(n_line)), dtype=torch.float32)
    v_or     = torch.as_tensor(getattr(obs, "v_or",   np.zeros(n_line)), dtype=torch.float32)
    theta_or = torch.as_tensor(getattr(obs, "theta_or",np.zeros(n_line)), dtype=torch.float32)

    p_ex     = torch.as_tensor(getattr(obs, "p_ex",   np.zeros(n_line)), dtype=torch.float32)
    q_ex     = torch.as_tensor(getattr(obs, "q_ex",   np.zeros(n_line)), dtype=torch.float32)
    v_ex     = torch.as_tensor(getattr(obs, "v_ex",   np.zeros(n_line)), dtype=torch.float32)
    theta_ex = torch.as_tensor(getattr(obs, "theta_ex",np.zeros(n_line)), dtype=torch.float32)

    rho      = torch.as_tensor(getattr(obs, "rho", np.zeros(n_line)), dtype=torch.float32)
    status   = torch.as_tensor(getattr(obs, "line_status", np.ones(n_line)), dtype=torch.float32)
    cooldown = torch.as_tensor(getattr(obs, "time_before_cooldown_line", np.zeros(n_line)), dtype=torch.float32)

    # ---------- Node features ----------
    # Allocate [N, d]
    d = 11
    x = torch.zeros((N, d), dtype=torch.float32)

    # helper to scatter values into x[:, channel] at specific topo positions
    def put(channel_tensor, topo_positions, feature_channel_idx):
        if len(topo_positions) == 0 or channel_tensor.numel() == 0:
            return
        idx = torch.as_tensor(topo_positions, dtype=torch.long)
        x[idx, feature_channel_idx] = channel_tensor

    # p, q, v, theta for gens/loads
    put(gen_p,     pos_gen, 0)
    put(gen_q,     pos_gen, 1)
    put(gen_v,     pos_gen, 2)
    put(gen_theta, pos_gen, 3)

    put(load_p,     pos_load, 0)
    put(load_q,     pos_load, 1)
    put(load_v,     pos_load, 2)
    put(load_theta, pos_load, 3)

    # line endpoints get both ends’ values
    put(p_or,     pos_or, 0); put(q_or,     pos_or, 1); put(v_or,     pos_or, 2); put(theta_or, pos_or, 3)
    put(p_ex,     pos_ex, 0); put(q_ex,     pos_ex, 1); put(v_ex,     pos_ex, 2); put(theta_ex, pos_ex, 3)

    # rho/status/cooldown mirrored to both endpoints
    put(rho,      pos_or, 4); put(rho,      pos_ex, 4)
    put(status,   pos_or, 5); put(status,   pos_ex, 5)
    put(cooldown, pos_or, 6); put(cooldown, pos_ex, 6)

    # danger flag at endpoints
    dang = (rho >= danger_thresh).float()
    put(dang, pos_or, 7); put(dang, pos_ex, 7)

    # type indicators
    x[torch.as_tensor(pos_gen,  dtype=torch.long),  8] = 1.0  # is_gen
    x[torch.as_tensor(pos_load, dtype=torch.long),  9] = 1.0  # is_load
    x[torch.as_tensor(np.concatenate([pos_or, pos_ex]), dtype=torch.long), 10] = 1.0  # is_line_endpoint

    # ---------- Edge index (lines, both directions) ----------
    u = torch.as_tensor(pos_or, dtype=torch.long)  # [n_line]
    v = torch.as_tensor(pos_ex, dtype=torch.long)  # [n_line]

    # base edges: or->ex and ex->or (directed both ways -> undirected in practice)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)  # [2, 2*n_line]
    edge_attr_list = []

    # edge attributes for both directions
    # forward (or -> ex)
    ea_fwd = torch.stack([rho, p_or, q_or, status, torch.ones_like(rho)], dim=1)  # [n_line, 5]
    # backward (ex -> or)
    ea_bwd = torch.stack([rho, p_ex, q_ex, status, -torch.ones_like(rho)], dim=1) # [n_line, 5]
    edge_attr = torch.cat([ea_fwd, ea_bwd], dim=0)  # [2*n_line, 5]

    # ---------- Optional: busbar adjacency ----------
    if include_busbar_adj:
        # Build from connectivity matrix (0/1). Only add edges not already in line edges.
        A = torch.as_tensor(obs.connectivity_matrix(), dtype=torch.float32)
        # keep strict upper triangle to list pairs once
        iu = torch.triu(A, diagonal=1).nonzero(as_tuple=False)  # [E_bus, 2]
        if iu.numel() > 0:
            bu = iu[:, 0].long()
            bv = iu[:, 1].long()
            bus_edges = torch.stack([torch.cat([bu, bv]), torch.cat([bv, bu])], dim=0)  # [2, 2*E_bus]
            # Append with zero edge_attr (or make something meaningful if you want)
            edge_index = torch.cat([edge_index, bus_edges], dim=1)
            zeros_attr = torch.zeros((2 * iu.size(0), edge_attr.size(1)), dtype=torch.float32)
            edge_attr = torch.cat([edge_attr, zeros_attr], dim=0)

    # ---------- Optional: self-loops ----------
    if add_self_loops:
        ii = torch.arange(N, dtype=torch.long)
        self_loops = torch.stack([ii, ii], dim=0)         # [2, N]
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        # zero attrs for self-loops
        edge_attr = torch.cat([edge_attr, torch.zeros((N, edge_attr.size(1)), dtype=torch.float32)], dim=0)

    # ---------- To device & pack ----------
    data = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),   # shape [E, 5]
        # You can also attach any global scalars you want, e.g.:
        # y=None, pos=None, etc.
    )
    return data