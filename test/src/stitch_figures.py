"""
Stitch figures across parameter sweeps into a single montage image.

Grid layout:
  rows   = (nonzero_alpha, flow_type) in fixed order:
           [('evenly','bidirectional'),
            ('evenly','outward'),
            ('center','bidirectional'),
            ('center','outward')]
  cols   = values of the chosen varying parameter (strength / alpha / Ni)
  cell   = a figure file inside:
           data/naive_simulation/fig/{flow_prefix}nonzero_alpha_{na}_fr_{strength}_agents_{agents}_N_i_{Ni}_alpha_{alpha}/{figure_name}

Usage examples:
  python stitch_figures.py --param_to_change strength
  python stitch_figures.py --param_to_change alpha --figure_name mean_cosine_similarity_heatmap.png
  python stitch_figures.py --param_to_change Ni --output out/combined_Ni.png
"""

import os
import glob
import math
import argparse
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Defaults (aligned to your runs)
# -----------------------------
DEFAULT_COUPLING_STRENGTHS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
DEFAULT_ALPHA_PER_DATA     = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
DEFAULT_N_I_LIST           = [25, 50, 100, 200, 400]
#DEFAULT_N_I_LIST           = [24, 49, 99, 199, 399]
DEFAULT_FLOW_TYPES         = ["bidirectional", "outward"]
DEFAULT_NONZERO_ALPHAS     = ["evenly", "center"]

# Fixed defaults (same as your scripts)
DEFAULT_AGENTS_COUNT = 7
DEFAULT_MAX_T = 1_000_000

# The most "tile-friendly" figure (no colorbar, square cells)
DEFAULT_FIGURE_NAME = "mean_distance_heatmap_Blues_no_colorbar.png"

ROW_ORDER = [
    ("evenly", "bidirectional"),
    ("evenly", "outward"),
    ("center", "bidirectional"),
    ("center", "outward"),
]

def get_subdir(flow_type: str, nonzero_alpha: str,
               coupling_strength: float, agents_count: int,
               N_i: int, alpha: float) -> str:
    """Mirror plot script's directory naming."""
    if flow_type == 'bidirectional':
        flow_prefix = 'bidirectional_flow-'
    elif flow_type == 'outward':
        flow_prefix = 'outward_flow-'
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    return f"{flow_prefix}nonzero_alpha_{nonzero_alpha}_fr_{coupling_strength}_agents_{agents_count}_N_i_{N_i}_alpha_{alpha}"

def try_load_or_placeholder(path: str, cell_w: int = 512, cell_h: int = 512) -> Image.Image:
    """Open image if available else return a placeholder with 'missing' text."""
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
            return img
        except Exception:
            pass
    # Placeholder
    img = Image.new("RGBA", (cell_w, cell_h), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img)
    msg = "missing"
    draw.text((10, 10), msg, fill=(160, 0, 0, 255))
    truncated = path[-80:] if len(path) > 80 else path
    draw.text((10, 30), truncated, fill=(80, 80, 80, 255))
    return img

def parse_list_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description="Stitch generated figures into a montage.")
    p.add_argument("--param_to_change", "-p",
                   choices=["strength", "alpha", "Ni", "flow_center"],
                   required=False,
                   help="Which parameter goes on the horizontal axis. "
                        "If omitted, all (strength, alpha, Ni) will be processed. "
                        "Use 'flow_center' for 4-panel layout with fixed parameters.")
    p.add_argument("--figure_name", default=DEFAULT_FIGURE_NAME,
                   help="File name (not glob) of the figure to stitch from each directory.")
    # Fixed values (used for the non-varying parameters)
    p.add_argument("--coupling_strength", type=float, default=0.01,
                   help="Fixed coupling strength if not varying.")
    p.add_argument("--alpha_per_data", type=float, default=0.001,
                   help="Fixed alpha_per_data if not varying.")
    p.add_argument("--N_i", type=int, default=100,
                   help="Fixed N_i if not varying.")
    p.add_argument("--agents_count", type=int, default=DEFAULT_AGENTS_COUNT)

    # Value lists (used when that param is chosen to vary)
    p.add_argument("--strength_values", type=parse_list_floats,
                   default=DEFAULT_COUPLING_STRENGTHS,
                   help="Comma-separated strengths for sweep (used when param_to_change=strength).")
    p.add_argument("--alpha_values", type=parse_list_floats,
                   default=DEFAULT_ALPHA_PER_DATA,
                   help="Comma-separated alphas for sweep (used when param_to_change=alpha).")
    p.add_argument("--Ni_values", type=parse_list_ints,
                   default=DEFAULT_N_I_LIST,
                   help="Comma-separated N_i values for sweep (used when param_to_change=Ni).")

    # In case you ever want to limit rows
    p.add_argument("--flow_types", default=",".join(DEFAULT_FLOW_TYPES),
                   help="Comma-separated subset of flow types (bidirectional,outward).")
    p.add_argument("--nonzero_alphas", default=",".join(DEFAULT_NONZERO_ALPHAS),
                   help="Comma-separated subset of nonzero_alpha (evenly,center).")

    # Layout / output
    p.add_argument("--cell_width", type=int, default=0,
                   help="Force cell width (0=auto from first image).")
    p.add_argument("--cell_height", type=int, default=0,
                   help="Force cell height (0=auto from first image).")
    p.add_argument("--margin", type=int, default=16, help="Outer margin (pixels).")
    p.add_argument("--gutter_x", type=int, default=12, help="Horizontal gap (pixels).")
    p.add_argument("--gutter_y", type=int, default=12, help="Vertical gap (pixels).")
    p.add_argument("--label_space_top", type=int, default=40,
                   help="Space reserved for top column labels.")
    p.add_argument("--label_space_left", type=int, default=160,
                   help="Space reserved for left row labels.")
    p.add_argument("--base_dir", default="data/naive_simulation/fig",
                   help="Root figure directory.")
    p.add_argument("--output", "-o", default="",
                   help="Output image path (default auto-generated).")

    # --- Overlay options ---
    p.add_argument("--overlay_on_tile", action="store_true",
                   help="Overlay text directly on top of each tile image.")
    p.add_argument("--overlay_fmt", default="{col_label}={val}",
                   help=(
                       "Python format for overlay text. Available keys: "
                       "col_label, val, nonzero_alpha, flow_type, strength, alpha, N_i"
                   ))
    p.add_argument("--overlay_bg_alpha", type=int, default=180,
                   help="Background box alpha (0-255) for overlay text.")
    p.add_argument("--overlay_pad", type=int, default=6,
                   help="Padding (px) inside the overlay background box.")
    p.add_argument("--hide_outer_labels", action="store_true",
                   help="Do not draw the outer top/left labels when set.")
    p.add_argument("--overlay_pos", choices=["tl", "tr", "bl", "br", "center"], default="tl",
                   help="Overlay position on each tile: top-left, top-right, bottom-left, bottom-right, or center.")
    p.add_argument("--suppress_redundant_overlay", action="store_true",
                   help="When outer labels are visible, drop row/column info from overlay to avoid duplication.")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode to check correspondence between row labels and figure paths.")
    return p

def debug_correspondence(args, target_param):
    """行ラベルと実際の図のパスの対応を確認"""
    import re

    # 行の設定
    chosen_flows = [s.strip() for s in args.flow_types.split(",") if s.strip()]
    chosen_nzas  = [s.strip() for s in args.nonzero_alphas.split(",") if s.strip()]
    row_pairs = [(na, ft) for (na, ft) in ROW_ORDER
                 if na in chosen_nzas and ft in chosen_flows]

    # 列の設定（最初の値のみチェック）
    if target_param == "strength":
        test_val = args.strength_values[0]
        col_label = "strength"
    elif target_param == "alpha":
        test_val = args.alpha_values[0]
        col_label = "alpha"
    elif target_param == "Ni":
        test_val = args.Ni_values[0]
        col_label = "N_i"

    print(f"\n=== 対応関係チェック (param: {target_param}, test_val: {test_val}) ===")

    def resolve_dir(na, ft, val):
        strength = args.coupling_strength
        alpha    = args.alpha_per_data
        Ni       = args.N_i
        if target_param == "strength":
            strength = val
        elif target_param == "alpha":
            alpha = val
        elif target_param == "Ni":
            Ni = int(val)

        subdir = get_subdir(ft, na, strength, args.agents_count, Ni, alpha)
        return os.path.join(args.base_dir, subdir)

    # 全ての列値を取得
    if target_param == "strength":
        col_values = args.strength_values
    elif target_param == "alpha":
        col_values = args.alpha_values
    elif target_param == "Ni":
        col_values = args.Ni_values

    # Check correspondence for each cell
    for ri, (na, ft) in enumerate(row_pairs):
        for ci, val in enumerate(col_values):
            fig_dir = resolve_dir(na, ft, val)
            fig_path = os.path.join(fig_dir, args.figure_name)
            exists = os.path.exists(fig_path)

            if not exists:
                print(f"Missing: Row {ri+1}, Col {ci+1} ({na}, {ft}, {col_label}={val})")
                print(f"  Path: {fig_path}")

def main():
    args = build_parser().parse_args()

    def run_one(target_param: str):
        # Prepare rows
        chosen_flows = [s.strip() for s in args.flow_types.split(",") if s.strip()]
        chosen_nzas  = [s.strip() for s in args.nonzero_alphas.split(",") if s.strip()]

        row_pairs = [(na, ft) for (na, ft) in ROW_ORDER
                     if na in chosen_nzas and ft in chosen_flows]
        if not row_pairs:
            raise SystemExit("No rows to build (check --flow_types and --nonzero_alphas).")

        # Prepare columns (varying values)
        if target_param == "strength":
            col_values = args.strength_values
            col_label = "strength"
        elif target_param == "alpha":
            col_values = args.alpha_values
            col_label = "alpha"
        elif target_param == "Ni":
            col_values = args.Ni_values
            col_label = "N_i"
        else:
            raise SystemExit(f"Unknown target_param: {target_param}")

        # Function to resolve the directory for a given (row, col)
        def resolve_dir(na, ft, val):
            strength = args.coupling_strength
            alpha    = args.alpha_per_data
            Ni       = args.N_i
            if target_param == "strength":
                strength = val
            elif target_param == "alpha":
                alpha = val
            elif target_param == "Ni":
                Ni = int(val)

            subdir = get_subdir(ft, na, strength, args.agents_count, Ni, alpha)
            return os.path.join(args.base_dir, subdir)

        # Load first valid image to determine cell size (if not forced)
        sample_img = None
        sample_w = args.cell_width or 0
        sample_h = args.cell_height or 0
        for na, ft in row_pairs:
            for v in col_values:
                candidate_dir = resolve_dir(na, ft, v)
                candidate_path = os.path.join(candidate_dir, args.figure_name)
                if os.path.exists(candidate_path):
                    try:
                        sample_img = Image.open(candidate_path).convert("RGBA")
                        sample_w = sample_w or sample_img.width
                        sample_h = sample_h or sample_img.height
                        break
                    except Exception:
                        pass
            if sample_img:
                break
        if sample_w == 0 or sample_h == 0:
            # Fallback if nothing found
            sample_w, sample_h = 512, 512

        cols = len(col_values)
        rows = len(row_pairs)

        W = (args.margin * 2 +
             args.label_space_left +
             cols * sample_w +
             (cols - 1) * args.gutter_x)
        H = (args.margin * 2 +
             args.label_space_top +
             rows * sample_h +
             (rows - 1) * args.gutter_y)

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Try to find a readable font
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        except Exception:
            print("Warning: Could not load custom font, using default.")
            font = ImageFont.load_default()

        # Label styles
        label_color = (20, 20, 20, 255)

        # Top column labels (optional)
        if not args.hide_outer_labels:
            for ci, val in enumerate(col_values):
                x = (args.margin + args.label_space_left +
                     ci * (sample_w + args.gutter_x))
                y = args.margin
                txt = f"{col_label} = {val}"
                draw.text((x, y), txt, fill=label_color, font=font)

        # Left row labels and tiles
        for ri, (na, ft) in enumerate(row_pairs):
            # Row label (optional)
            y_label = (args.margin + args.label_space_top +
                       ri * (sample_h + args.gutter_y))
            if not args.hide_outer_labels:
                row_label = f"{na}\n{ft}"
                draw.multiline_text((args.margin, y_label), row_label, fill=label_color, font=font, spacing=4)

            # Helper functions for overlay
            def build_overlay_text(val, na, ft):
                """Return overlay text string respecting redundancy suppression and fmt."""
                strength_val = args.coupling_strength if target_param != "strength" else val
                alpha_val    = args.alpha_per_data   if target_param != "alpha"    else val
                Ni_val       = args.N_i              if target_param != "Ni"       else int(val)

                # Base dict
                fields = {
                    "col_label": col_label,
                    "val": val,
                    "nonzero_alpha": na,
                    "flow_type": ft,
                    "strength": str(strength_val),
                    "alpha": str(alpha_val),
                    "N_i": str(Ni_val),
                }
                # If suppressing duplicates and outer labels are shown, remove row/col info
                if args.suppress_redundant_overlay and not args.hide_outer_labels:
                    for k in ("col_label", "nonzero_alpha", "flow_type"):
                        fields.pop(k, None)
                    # Safe field substitution
                    safe_fields = {k: (fields.get(k, "")) for k in [
                        "col_label", "val", "nonzero_alpha", "flow_type", "strength", "alpha", "N_i"]}
                    return args.overlay_fmt.format(**safe_fields)
                return args.overlay_fmt.format(**fields)

            def compute_overlay_position(text_w, text_h, sample_w, sample_h, pad):
                if args.overlay_pos == "tl":
                    return pad, pad
                if args.overlay_pos == "tr":
                    return max(0, sample_w - text_w - 2 * pad), pad
                if args.overlay_pos == "bl":
                    return pad, max(0, sample_h - text_h - 2 * pad)
                if args.overlay_pos == "br":
                    return max(0, sample_w - text_w - 2 * pad), max(0, sample_h - text_h - 2 * pad)
                # center
                return max(0, (sample_w - text_w) // 2 - pad), max(0, (sample_h - text_h) // 2 - pad)

            # Cells
            for ci, val in enumerate(col_values):
                x = (args.margin + args.label_space_left +
                     ci * (sample_w + args.gutter_x))
                y = (args.margin + args.label_space_top +
                     ri * (sample_h + args.gutter_y))

                fig_dir = resolve_dir(na, ft, val)
                fig_path = os.path.join(fig_dir, args.figure_name)
                tile = try_load_or_placeholder(fig_path, sample_w, sample_h)

                # resize if forced cell size
                if tile.size != (sample_w, sample_h):
                    tile = tile.resize((sample_w, sample_h), Image.BILINEAR)

                # Optional overlay text on the tile itself
                if args.overlay_on_tile:
                    overlay_text = build_overlay_text(val, na, ft)
                    tile_draw = ImageDraw.Draw(tile)
                    try:
                        bbox = tile_draw.textbbox((0, 0), overlay_text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    except Exception:
                        text_w, text_h = tile_draw.textlength(overlay_text, font=font), 18

                    pad = args.overlay_pad
                    ox, oy = compute_overlay_position(text_w, text_h, sample_w, sample_h, pad)
                    bg_w = min(sample_w - ox, text_w + 2 * pad)
                    bg_h = min(sample_h - oy, text_h + 2 * pad)

                    # Semi-transparent background rectangle at computed position
                    bg = Image.new("RGBA", (bg_w, bg_h), (255, 255, 255, args.overlay_bg_alpha))
                    tile.alpha_composite(bg, dest=(ox, oy))

                    # Draw text over the background on the tile
                    tile_draw.text((ox + pad, oy + pad), overlay_text, fill=label_color, font=font)

                # Finally paste the (possibly annotated) tile
                canvas.paste(tile, (x, y), tile)

        # Output path handling
        if args.output:
            base_out = args.output
            root, ext = os.path.splitext(base_out)
            if ext == "":
                ext = ".png"
            if multiple_runs:
                out_path = f"{root}_{col_label}{ext}"
            else:
                out_path = f"{root}{ext}"
        else:
            # Auto-generated path structure
            safe_fig = os.path.splitext(os.path.basename(args.figure_name))[0]
            fixed_dir = f"fixed_agents_{args.agents_count}_Ni_{args.N_i}_alpha_{args.alpha_per_data}_fr_{args.coupling_strength}"

            out_path = os.path.join(
                args.base_dir, "montage", "param_sweeps",
                fixed_dir, safe_fig, f"{col_label}.png"
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        canvas.save(out_path)
        print(f"[OK] Saved: {out_path}")

    def run_flow_center_comparison():
        """Create 4-panel comparison: flow_type (bidirectional/outward) × nonzero_alpha (center/evenly)"""
        # Fixed 2x2 grid
        row_pairs = [
            ("evenly", "bidirectional"),
            ("evenly", "outward"),
            ("center", "bidirectional"),
            ("center", "outward"),
        ]

        # All parameters are fixed
        strength = args.coupling_strength
        alpha = args.alpha_per_data
        Ni = args.N_i

        def resolve_dir(na, ft):
            subdir = get_subdir(ft, na, strength, args.agents_count, Ni, alpha)
            return os.path.join(args.base_dir, subdir)

        # Load first valid image to determine cell size
        sample_img = None
        sample_w = args.cell_width or 0
        sample_h = args.cell_height or 0
        for na, ft in row_pairs:
            candidate_dir = resolve_dir(na, ft)
            candidate_path = os.path.join(candidate_dir, args.figure_name)
            if os.path.exists(candidate_path):
                try:
                    sample_img = Image.open(candidate_path).convert("RGBA")
                    sample_w = sample_w or sample_img.width
                    sample_h = sample_h or sample_img.height
                    break
                except Exception:
                    pass
        if sample_w == 0 or sample_h == 0:
            sample_w, sample_h = 512, 512

        # 2x2 grid
        cols, rows = 2, 2
        W = (args.margin * 2 +
             args.label_space_left +
             cols * sample_w +
             (cols - 1) * args.gutter_x)
        H = (args.margin * 2 +
             args.label_space_top +
             rows * sample_h +
             (rows - 1) * args.gutter_y)

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Font setup
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        except Exception:
            font = ImageFont.load_default()
        label_color = (20, 20, 20, 255)

        # Column labels (nonzero_alpha)
        if not args.hide_outer_labels:
            for ci, nonzero_alpha in enumerate(["evenly", "center"]):
                x = (args.margin + args.label_space_left +
                     ci * (sample_w + args.gutter_x))
                y = args.margin
                draw.text((x, y), nonzero_alpha, fill=label_color, font=font)

        # Row labels and tiles
        for ri, flow_type in enumerate(["bidirectional", "outward"]):
            # Row label
            y_label = (args.margin + args.label_space_top +
                       ri * (sample_h + args.gutter_y))
            if not args.hide_outer_labels:
                draw.text((args.margin, y_label), flow_type, fill=label_color, font=font)

            # Cells for this row
            for ci, nonzero_alpha in enumerate(["evenly", "center"]):
                x = (args.margin + args.label_space_left +
                     ci * (sample_w + args.gutter_x))
                y = (args.margin + args.label_space_top +
                     ri * (sample_h + args.gutter_y))

                fig_dir = resolve_dir(nonzero_alpha, flow_type)
                fig_path = os.path.join(fig_dir, args.figure_name)
                tile = try_load_or_placeholder(fig_path, sample_w, sample_h)

                if tile.size != (sample_w, sample_h):
                    tile = tile.resize((sample_w, sample_h), Image.BILINEAR)

                # Optional overlay
                if args.overlay_on_tile:
                    overlay_text = f"{nonzero_alpha}\n{flow_type}"
                    tile_draw = ImageDraw.Draw(tile)
                    try:
                        font = ImageFont.load_default()  # Fallback for overlay
                        bbox = tile_draw.textbbox((0, 0), overlay_text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    except Exception:
                        text_w, text_h = 50, 36  # Fallback dimensions

                    pad = args.overlay_pad
                    ox, oy = pad, pad  # Simple top-left positioning
                    bg_w = min(sample_w - ox, text_w + 2 * pad)
                    bg_h = min(sample_h - oy, text_h + 2 * pad)

                    bg = Image.new("RGBA", (bg_w, bg_h), (255, 255, 255, args.overlay_bg_alpha))
                    tile.alpha_composite(bg, dest=(ox, oy))
                    tile_draw.multiline_text((ox + pad, oy + pad), overlay_text, fill=label_color, font=font)

                canvas.paste(tile, (x, y), tile)

        # Output path for flow_center comparison
        safe_fig = os.path.splitext(os.path.basename(args.figure_name))[0]
        out_path = os.path.join(
            args.base_dir, "montage", "comparisons",
            f"fixed_agents_{args.agents_count}_Ni_{Ni}_alpha_{alpha}_fr_{strength}",
            safe_fig,
            "flow_center.png"
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        canvas.save(out_path)
        print(f"[OK] Saved: {out_path}")

    # Decide which parameters to run
    if args.param_to_change:
        if args.param_to_change == "flow_center":
            # Special case: run 4-panel flow/center comparison
            if args.debug:
                print("Debug mode enabled but flow_center comparison doesn't use parameter sweeps")
            run_flow_center_comparison()
            return
        else:
            params_to_run = [args.param_to_change]
    else:
        # When omitted, run all three sweeps in the order: Ni, strength, alpha
        params_to_run = ["Ni", "strength", "alpha"]

    # Flag for output naming inside run_one
    global multiple_runs
    multiple_runs = (len(params_to_run) > 1)

    for p in params_to_run:
        if args.debug:
            debug_correspondence(args, p)
        run_one(p)

if __name__ == "__main__":
    main()