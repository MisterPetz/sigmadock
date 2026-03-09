import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import torch

from sigmadock.chem import EDGE_ENTITY, NODE_ENTITY


def _plot_molecule(
    pos: np.ndarray,
    fi: np.ndarray,
    fc: np.ndarray,
    protein_nodes: np.ndarray,
    ligand_atom_nodes: np.ndarray,
    protein_virtual: np.ndarray,
    ligand_virtual: np.ndarray,
    overconstrained_dummies: torch.Tensor,
    show_protein: bool,
    show_ligand_virtual: bool,
    show_protein_virtual: bool,
    show_overconstrained: bool,
    palette: list[str],
    name_prefix: str = "",
    opacity: float = 1.0,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Add atomic/bead markers to `fig`. Legend entries kept minimal and meaningful."""
    if fig is None:
        fig = go.Figure()

    # protein atoms: single legend entry
    if show_protein and len(protein_nodes) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pos[protein_nodes, 0],
                y=pos[protein_nodes, 1],
                z=pos[protein_nodes, 2],
                mode="markers",
                marker={
                    "size": 6,
                    "color": "rgb(190,190,190)",
                    "symbol": "circle",
                    "opacity": 0.5 * opacity,
                    "line": {"width": 0.4, "color": "rgba(0,0,0,0.08)"},
                },
                name=f"{name_prefix}Protein atoms",
                legendgroup="protein",
                showlegend=True,
            )
        )

    # ligand fragments: draw all fragments and SHOW a legend entry for each fragment
    unique_frags = np.unique(fc[ligand_atom_nodes]) if len(ligand_atom_nodes) > 0 else np.array([])
    for idx, frag in enumerate(unique_frags):
        color = palette[idx % len(palette)]
        lig_idxs = fi[ligand_atom_nodes][np.where(fc[ligand_atom_nodes] == frag)[0]]
        if not show_overconstrained:
            is_overconstrained = overconstrained_dummies[lig_idxs]
            idxs = lig_idxs[is_overconstrained == 0] + len(protein_nodes)
        else:
            idxs = lig_idxs + len(protein_nodes)

        # *** Show legend entry for each fragment so they are individually togglable ***
        fig.add_trace(
            go.Scatter3d(
                x=pos[idxs, 0],
                y=pos[idxs, 1],
                z=pos[idxs, 2],
                mode="markers",
                marker={
                    "size": 6,
                    "color": color,
                    "symbol": "circle",
                    "opacity": 0.95 * opacity,
                    "line": {"width": 1.0, "color": "rgba(0,0,0,0.18)"},
                },
                name=f"{name_prefix}Sample lig frag {frag}",
                # no shared legendgroup — each fragment acts independently
                showlegend=True,
            )
        )

    # overconstrained / special nodes: one legend entry
    if show_overconstrained:
        over_idx = torch.where(overconstrained_dummies == 1)[0] + len(protein_nodes)
        if len(over_idx) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=pos[over_idx, 0].flatten(),
                    y=pos[over_idx, 1].flatten(),
                    z=pos[over_idx, 2].flatten(),
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": "rgb(220,40,40)",
                        "symbol": "cross",
                        "opacity": 1.0,
                        "line": {"width": 1.2, "color": "rgba(0,0,0,0.25)"},
                    },
                    name=f"{name_prefix}Overconstrained",
                    legendgroup="special",
                    showlegend=True,
                )
            )

    # virtual nodes (one legend entry each)
    if show_protein_virtual and len(protein_virtual) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pos[protein_virtual, 0],
                y=pos[protein_virtual, 1],
                z=pos[protein_virtual, 2],
                mode="markers",
                marker={
                    "size": 8,
                    "color": "rgb(200,40,40)",
                    "symbol": "diamond",
                    "opacity": 0.9,
                    "line": {"width": 1.5, "color": "rgba(0,0,0,0.18)"},
                },
                name=f"{name_prefix}Protein virtual",
                legendgroup="virtual",
                showlegend=True,
            )
        )
    if show_ligand_virtual and len(ligand_virtual) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pos[ligand_virtual, 0],
                y=pos[ligand_virtual, 1],
                z=pos[ligand_virtual, 2],
                mode="markers",
                marker={
                    "size": 8,
                    "color": "rgb(40,40,200)",
                    "symbol": "diamond",
                    "opacity": 0.8,
                    "line": {"width": 1.5, "color": "rgba(0,0,0,0.18)"},
                },
                name=f"{name_prefix}Ligand virtual",
                legendgroup="virtual",
                showlegend=True,
            )
        )

    return fig


def _add_edges(  # noqa: C901
    fig: go.Figure,
    ei: np.ndarray,
    ee: np.ndarray,
    pos: np.ndarray,
    protein_virtual: np.ndarray,
    ligand_virtual: np.ndarray,
    fi: np.ndarray,
    fc: np.ndarray,
    ligand_atom_nodes: np.ndarray,
    overconstrained_dummies: np.ndarray,
    show_protein: bool,
    show_protein_virtual: bool,
    show_ligand_virtual: bool,
    show_protein_ligand_virtual: bool,
    show_triangulation: bool,
    show_overconstrained: bool,
    show_interaction_edges: bool,
    name_prefix: str = "",
    opacity: float = 1.0,
    palette: list[str] | None = None,
) -> None:
    """
    Add edges (bonds/triangulation/virtuals). Low-level traces are hidden from legend;
    a single representative legend entry is shown for triangulation and interactions.
    """
    edge_coords = {k: [] for k in ["pp", "pl", "ll", "tri", "frag", "pli", "fi", "other"]}
    edge_frags = []
    overconstrained_lignodes = ligand_atom_nodes[np.where(overconstrained_dummies == 1)[0]]

    for e_idx, (i, j) in enumerate(ei.T):
        cat = "other"
        # Do not ad torsional bond cause this is tri-edge.
        if ee[e_idx] == EDGE_ENTITY["ligand_anchor_dummy"]:
            continue
        # filters
        if not show_protein and ee[e_idx] == EDGE_ENTITY["protein_bonds"]:
            continue
        if (i in ligand_virtual and j in ligand_virtual) and not show_ligand_virtual:
            continue
        if not show_protein_ligand_virtual and (
            (i in ligand_virtual and j in protein_virtual) or (j in ligand_virtual and i in protein_virtual)
        ):
            continue
        if ee[e_idx] == EDGE_ENTITY["protein_v2v"] and not show_protein_virtual:
            continue
        if ee[e_idx] == EDGE_ENTITY["fragment_triangulation"] and not show_triangulation:
            continue
        if not show_overconstrained and ((i in overconstrained_lignodes) or (j in overconstrained_lignodes)):
            continue

        # virtual edges categories
        if i in protein_virtual and j in protein_virtual:
            cat = "pp"
        elif (i in protein_virtual and j in ligand_virtual) or (i in ligand_virtual and j in protein_virtual):
            cat = "pl"
        elif i in ligand_virtual and j in ligand_virtual:
            cat = "ll"

        # other edges classification
        if ee[e_idx] == EDGE_ENTITY["fragment_triangulation"]:
            cat = "tri"
        elif (
            ee[e_idx] == EDGE_ENTITY["ligand_bonds"]
            or ee[e_idx] == EDGE_ENTITY["ligand_torsional_bond"]
            or ee[e_idx] == EDGE_ENTITY["ligand_anchor_dummy"]
        ):
            cat = "frag"
            frag_i = fc[fi == i - len(fi[fi < 0])][0] if i in ligand_atom_nodes else None
            edge_frags.append(frag_i)
        elif ee[e_idx] == EDGE_ENTITY["inter_complex"]:
            if not show_interaction_edges:
                continue
            cat = "pli"
        elif ee[e_idx] == EDGE_ENTITY["inter_fragments"]:
            if not show_interaction_edges:
                continue
            cat = "fi"

        edge_coords[cat].append((i, j))

    # fallback palette for edges (if palette not provided)
    fallback = ["#9E9E9E", "#7B9CE1", "#8CCB9B", "#E29AC2", "#C38C6D", "#A570C7"]

    legend_names = {
        "pp": "PP virtual",
        "pl": "P-L virtual",
        "ll": "LL virtual",
        "tri": "Triangulation",
        "frag": "Ligand bonds",
        "pli": "P-L Interaction",
        "fi": "Frag Interaction",
        "other": "Protein bonds",
    }

    col_idx = 0
    for cat, pairs in edge_coords.items():
        if len(pairs) == 0:
            col_idx += 1
            continue

        if cat == "frag":
            # ligand bonds: many traces (hidden in legend), colored by fragment
            for (i, j), frag in zip(pairs, edge_frags):
                xs = [pos[i, 0], pos[j, 0], None]
                ys = [pos[i, 1], pos[j, 1], None]
                zs = [pos[i, 2], pos[j, 2], None]
                color = (palette[frag % len(palette)]) if palette is not None else fallback[col_idx % len(fallback)]
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line={"color": color, "width": 6},
                        opacity=0.95 * opacity,
                        name=f"{name_prefix}frag_{frag}",
                        showlegend=False,
                        hoverinfo="none",
                    )
                )
        else:
            xs, ys, zs = [], [], []
            for i, j in pairs:
                xs += [pos[i, 0], pos[j, 0], None]
                ys += [pos[i, 1], pos[j, 1], None]
                zs += [pos[i, 2], pos[j, 2], None]

            # choose category styling; decide whether to show a single legend entry
            if cat in ("pp", "pl", "ll"):
                line_w = 1.8
                line_op = 0.32
                color = "rgba(120,120,160,1.0)"
                showlegend = False
            elif cat == "tri":
                # make triangulation clearly visible: orange, thicker, more opaque
                line_w = 3.2
                line_op = 0.92
                color = "orange"
                showlegend = True
            elif cat in ("pli", "fi"):
                line_w = 2.8
                line_op = 0.88
                color = "#D55E5E"
                showlegend = True
            else:
                line_w = 2.5
                line_op = 0.3
                color = "rgba(100,100,100,1.0)"
                showlegend = False

            # show one legend entry for categories marked showlegend=True
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line={"color": color, "width": line_w},
                    opacity=line_op * opacity,
                    name=f"{name_prefix}{legend_names[cat]}",
                    showlegend=showlegend,
                    hoverinfo="none",
                )
            )
        col_idx += 1


def add_sphere(
    fig: go.Figure,
    center: np.array,
    radius: float,
    color: str = "gold",
    opacity: float = 0.55,
    resolution: int = 20,
    name: str = "sphere",
) -> go.Figure:
    """Add a glossy-looking sphere surface (CoM)."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    surfacecolor = np.full_like(x, 0.5)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surfacecolor,
            colorscale=[[0, color], [1, color]],
            cmin=0,
            cmax=1,
            showscale=False,
            opacity=opacity,
            name=name,
            hoverinfo="skip",
            lighting={"ambient": 0.7, "diffuse": 0.6, "specular": 0.5, "roughness": 0.4, "fresnel": 0.1},
            lightposition={"x": 100, "y": 200, "z": 0},
            contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        )
    )
    return fig


def plot_interaction_graph_3d_plotly(  # noqa: C901
    graph: dict,
    show_protein: bool = True,
    show_ligand_virtual: bool = False,
    show_ligand_virtual_edges: bool = False,
    show_protein_virtual: bool = False,
    show_protein_virtual_edges: bool = False,
    show_protein_ligand_virtual_edges: bool = False,
    show_overconstrained: bool = False,
    show_triangulation: bool = False,
    show_interaction_edges: bool = False,
    pos_key: str = "pos_0",
    pocket_com: str | torch.Tensor | np.ndarray | None = None,
    ligand_com: str | torch.Tensor | np.ndarray | None = None,
    com_radius: float = 1.0,
    ref_pos_key: str | None = None,
    ref_opacity: float = 0.3,
    camera_angle: dict | None = None,
    fig: go.Figure = None,
) -> go.Figure:
    """
    Main plotting function. Returns a Plotly Figure. If fig is None it will show the figure.
    """
    should_show = fig is None
    if fig is None:
        fig = go.Figure()

    ei = graph["edge_index"].cpu().numpy()
    ee = graph["edge_entity"].cpu().numpy()
    ne = graph["node_entity"].cpu().numpy()
    fi = graph["frag_atom_idx"].cpu().numpy()
    fc = graph["frag_counter"].cpu().numpy()
    overconstrained_dummies = graph["overconstrained_dummies"]

    protein_nodes = np.where(fi < 0)[0]
    protein_virtual = np.where(ne == NODE_ENTITY["is_protein_virtual"])[0]
    ligand_virtual = np.where(ne == NODE_ENTITY["is_ligand_virtual"])[0]
    ligand_atom_nodes = np.where(
        (ne == NODE_ENTITY["is_ligand_atom"])
        | (ne == NODE_ENTITY["is_ligand_anchor"])
        | (ne == NODE_ENTITY["is_ligand_dummy"])
    )[0]

    palette = px.colors.qualitative.Dark24
    sample_pos = graph[pos_key].cpu().numpy()

    # 1) Add sample edges first (so beads render on top)
    _add_edges(
        fig,
        ei,
        ee,
        sample_pos,
        protein_virtual,
        ligand_virtual,
        fi,
        fc,
        ligand_atom_nodes,
        overconstrained_dummies,
        show_protein,
        show_protein_virtual_edges,
        show_ligand_virtual_edges,
        show_protein_ligand_virtual_edges,
        show_triangulation,
        show_overconstrained,
        show_interaction_edges,
        name_prefix="Sample ",
        opacity=1.0,
        palette=palette,
    )

    # 2) sample spheres (CoM) placed before beads but after edges
    if ligand_com is not None:
        if isinstance(ligand_com, str) and ligand_com in graph:
            ligand_com = graph[ligand_com].cpu().numpy()
        else:
            ligand_com = ligand_com.cpu().numpy() if isinstance(ligand_com, torch.Tensor) else ligand_com
        ligand_com = np.asarray(ligand_com).reshape(-1)
        add_sphere(fig, ligand_com, radius=com_radius, color="rgb(60,120,200)", opacity=0.45, name="Ligand CoM")

    if pocket_com is not None:
        if isinstance(pocket_com, str) and pocket_com in graph:
            pocket_com = graph[pocket_com].cpu().numpy()
        else:
            pocket_com = pocket_com.cpu().numpy() if isinstance(pocket_com, torch.Tensor) else pocket_com
        pocket_com = np.asarray(pocket_com).reshape(-1)
        add_sphere(fig, pocket_com, radius=com_radius, color="gold", opacity=0.40, name="Pocket CoM")

    # 3) sample beads (atoms)
    fig = _plot_molecule(
        sample_pos,
        fi,
        fc,
        protein_nodes,
        ligand_atom_nodes,
        protein_virtual,
        ligand_virtual,
        overconstrained_dummies,
        show_protein,
        show_ligand_virtual,
        show_protein_virtual,
        show_overconstrained,
        palette,
        name_prefix="Sample ",
        opacity=1.0,
        fig=fig,
    )

    # 4) reference geometry (edges -> spheres -> beads) drawn beneath sample beads
    if ref_pos_key is not None:
        ref_pos = graph[ref_pos_key].cpu().numpy()

        # reference edges (draw under sample)
        _add_edges(
            fig,
            ei,
            ee,
            ref_pos,
            protein_virtual,
            ligand_virtual,
            fi,
            fc,
            ligand_atom_nodes,
            overconstrained_dummies,
            show_protein=False,
            show_protein_virtual=False,
            show_ligand_virtual=False,
            show_protein_ligand_virtual=False,
            show_triangulation=False,
            show_overconstrained=False,
            show_interaction_edges=False,
            name_prefix="Reference ",
            opacity=ref_opacity,
            palette=palette,
        )

        # reference spheres
        if ligand_com is not None:
            add_sphere(
                fig, ligand_com, radius=com_radius, color="rgba(60,120,200,0.3)", opacity=0.28, name="Ref Ligand CoM"
            )
        if pocket_com is not None:
            add_sphere(
                fig, pocket_com, radius=com_radius, color="rgba(200,150,30,0.3)", opacity=0.28, name="Ref Pocket CoM"
            )

        # add reference beads semi-transparent
        fig = _plot_molecule(
            ref_pos,
            fi,
            fc,
            protein_nodes,
            ligand_atom_nodes,
            protein_virtual,
            ligand_virtual,
            overconstrained_dummies,
            show_protein=False,
            show_ligand_virtual=False,
            show_protein_virtual=False,
            show_overconstrained=False,
            palette=palette,
            name_prefix="Reference ",
            opacity=ref_opacity,
            fig=fig,
        )

    # layout: tighter scene domain, compact legend
    fig.update_layout(
        width=820,
        height=520,
        scene={
            "domain": {"x": [0.0, 0.88], "y": [0.0, 1.0]},
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False, "showticklabels": False},
            "bgcolor": "white",
            "camera_eye": {"x": 0.9, "y": 0.9, "z": 0.9},
            "dragmode": "orbit",
        },
        scene_aspectmode="data",
        margin={"l": 6, "r": 6, "b": 6, "t": 32},
        paper_bgcolor="white",
        legend={
            "x": 0.94,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(0,0,0,0.06)",
            "borderwidth": 0.5,
            "font": {"size": 10},
            "itemclick": "toggle",  # SINGLE-CLICK hides/deselects the clicked item
            "itemdoubleclick": "toggleothers",
        },
        uirevision="persist_camera_and_legend",
    )
    fig.update_layout(uirevision=True, dragmode="orbit", scene={"aspectmode": "data"}).update_layout(
        # key option:
        dragmode="zoom"
    )
    if camera_angle:
        fig.update_layout(scene_camera=camera_angle)

    if should_show:
        fig.show(
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "hoverClosestCartesian", "hoverCompareCartesian"],
                "displaylogo": False,
                "responsive": True,
            }
        )

    return fig
