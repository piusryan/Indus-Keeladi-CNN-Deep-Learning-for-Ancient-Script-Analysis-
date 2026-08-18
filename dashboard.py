"""
Indus-Keeladi CNN Project Dashboard
Comprehensive UI displaying ALL evaluation results from evaluation_results/
auto-discovered and rendered in organized sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import json


EVAL_DIR = Path(__file__).parent / "models" / "evaluation_results"


def load_training_summary():
    """Load training summary information"""
    model_path = Path(__file__).parent / "models" / "indus_classifier.keras"
    class_names_path = Path(__file__).parent / "models" / "indus_classifier_classes.txt"

    summary = {
        "model_exists": model_path.exists(),
        "class_names": []
    }

    if class_names_path.exists():
        with open(class_names_path, 'r') as f:
            names = [l.strip() for l in f.read().split('\n') if l.strip()]
            summary["class_names"] = names

    return summary


def scan_evaluation_results(eval_dir: Path):
    """
    Auto-scan evaluation_results/ directory and organize everything into categories.
    Returns a dict with:
      report_txt: path object (or None)
      aggregate_pngs: list of paths
      gallery_pngs: list of paths (graffiti_gallery_*)
      comparison_pngs: list of paths (graffiti_vs_indus_*)
      other_pngs: list of paths
      all_pngs: list of paths
      counts: dict of totals
    """
    result = {
        "report_txt": None,
        "aggregate_pngs": [],
        "gallery_pngs": [],
        "comparison_pngs": [],
        "other_pngs": [],
        "all_pngs": [],
        "counts": {},
        "dir_exists": eval_dir.exists(),
    }

    if not eval_dir.exists():
        return result

    files = sorted(eval_dir.iterdir())

    for f in files:
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        if f.suffix.lower() == '.txt' and 'report' in name_lower:
            result["report_txt"] = f
        elif f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            result["all_pngs"].append(f)
            if name_lower.startswith("graffiti_gallery_"):
                result["gallery_pngs"].append(f)
            elif name_lower.startswith("graffiti_vs_indus_"):
                result["comparison_pngs"].append(f)
            elif name_lower in ("match_statistics.png", "confidence_distribution.png"):
                result["aggregate_pngs"].append(f)
            else:
                result["other_pngs"].append(f)

    result["counts"] = {
        "total_pngs": len(result["all_pngs"]),
        "aggregate": len(result["aggregate_pngs"]),
        "gallery_pages": len(result["gallery_pngs"]),
        "comparison_pages": len(result["comparison_pngs"]),
        "other": len(result["other_pngs"]),
    }

    return result


def _group_pages(paths):
    """Group multi-page PNGs like xyz_page01.png, xyz_page02.png by folder_key"""
    groups = defaultdict(list)
    for p in paths:
        name = p.stem
        # e.g. "graffiti_gallery_general_keeladi_graffiti_page01"
        # strip _pageXX suffix
        m = re.match(r'^(.*)_page\d+$', name, re.IGNORECASE)
        if m:
            group_key = m.group(1)
        else:
            group_key = name
        groups[group_key].append(p)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x.name)
    return dict(sorted(groups.items()))


def _friendly_group_name(group_key: str) -> str:
    """Turn a raw file stem into a clean section title"""
    k = group_key
    k = k.replace('graffiti_gallery_', '').replace('graffiti_vs_indus_', '')
    k = k.replace('_', ' ').strip()
    # Title case words but preserve folder labels
    return '  —  '.join(part.title() for part in k.split('  ')) or k


def parse_report_metrics(report_txt_path: Path):
    """Parse quick metrics (match rate, total images, etc.) out of the .txt report"""
    if not report_txt_path or not report_txt_path.exists():
        return {}
    text = report_txt_path.read_text(encoding='utf-8', errors='ignore')
    metrics = {}

    patterns = {
        "total_images": r'Total Images:\s+(\d+)',
        "total_keeladi": r'Total Keeladi Images:\s+(\d+)',
        "high_conf": r'High Confidence Matches.*?:\s+(\d+)',
        "match_rate": r'Match Rate:\s+([0-9.]+)%',
        "mean_conf": r'Mean Confidence:\s+([0-9.]+)',
        "unique_signs": r'Unique Indus Signs Matched:\s+(\d+)',
    }
    for label, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                metrics[label] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
            except ValueError:
                pass
    return metrics


def render_section_grid(images, cols=2, width_label=True):
    """Render a list of image paths in a Streamlit column grid"""
    if not images:
        return
    grid = st.columns(cols)
    for idx, img_path in enumerate(images):
        with grid[idx % cols]:
            caption = img_path.name
            if width_label:
                # Show file size + modified time
                try:
                    size_kb = img_path.stat().st_size / 1024.0
                    caption += f"  ·  {size_kb:.0f} KB"
                except Exception:
                    pass
            st.image(str(img_path), caption=caption, use_container_width=True)


def render_grouped_section(title, icon, grouped_paths, description, cols=1):
    """Render a top-level section with grouped, multi-page images"""
    st.markdown("---")
    st.header(f"{icon} {title}")

    if not grouped_paths:
        st.info("No images of this type found yet — run the evaluation pipeline first.")
        return

    total_pages = sum(len(v) for v in grouped_paths.values())
    st.caption(f"**{len(grouped_paths)} group(s) · {total_pages} page(s)  ·  {description}**")

    for group_key, pages in grouped_paths.items():
        name = _friendly_group_name(group_key)
        with st.expander(f"📂 {name}  ·  {len(pages)} page(s)", expanded=True):
            # If a group has many pages, put a selector; otherwise render all
            if len(pages) <= 6:
                render_section_grid(pages, cols=cols)
            else:
                pg = st.selectbox(
                    "Select page",
                    options=list(range(1, len(pages) + 1)),
                    format_func=lambda i: f"Page {i} / {len(pages)}  ·  {pages[i-1].name}",
                    key=f"sel-{group_key}",
                )
                st.image(str(pages[pg - 1]), caption=pages[pg - 1].name, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Indus-Keeladi CNN Project Dashboard",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar: quick navigation & result counts ───────────────────────
    scan = scan_evaluation_results(EVAL_DIR)
    counts = scan["counts"]
    metrics = parse_report_metrics(scan["report_txt"])
    training = load_training_summary()

    with st.sidebar:
        st.title("🏺 Indus-Keeladi")
        st.subheader("Evaluation Results")
        st.metric("Total PNG Artifacts", counts.get("total_pngs", 0))
        c1, c2 = st.columns(2)
        c1.metric("Aggregate Plots", counts.get("aggregate", 0))
        c2.metric("Gallery Pages", counts.get("gallery_pages", 0))
        c1.metric("Compare Pages", counts.get("comparison_pages", 0))
        c2.metric("Other Images", counts.get("other", 0))

        if metrics:
            st.markdown("---")
            st.subheader("Latest Run Metrics")
            if "match_rate" in metrics:
                st.metric("Overall Match Rate", f"{metrics['match_rate']}%")
            if "total_images" in metrics:
                st.metric("Images Analyzed", int(metrics["total_images"]))
            if "mean_conf" in metrics:
                st.metric("Mean Confidence", f"{metrics['mean_conf']:.3f}")
            if "unique_signs" in metrics:
                st.metric("Unique Indus Signs Hit", int(metrics["unique_signs"]))

        st.markdown("---")
        st.subheader("Source Folder")
        st.code(str(EVAL_DIR), language="text")
        st.markdown(
            f"Files regenerate every time you run the pipeline.\n"
            f"Everything you see below is **auto-discovered** from that folder — "
            f"no hardcoded image paths."
        )

    # ── Header ──────────────────────────────────────────────────────────
    st.title("🏺 Indus-Keeladi CNN Project Dashboard")
    st.markdown(
        "All evaluation artifacts generated by the pipeline are rendered below. "
        "New PNGs added to `models/evaluation_results/` will appear automatically on the next page load."
    )

    # ── 0. LIVE METRICS STRIP ───────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Project Overview")
    cols = st.columns(5)
    cols[0].metric("Model", "✅ Ready" if training["model_exists"] else "❌ Missing",
                   f"{len(training['class_names'])} Indus classes")
    cols[1].metric("Evaluation PNGs", counts.get("total_pngs", 0),
                   f"{counts.get('gallery_pages', 0) + counts.get('comparison_pages', 0)} graffiti pages")
    cols[2].metric("Match Rate",
                   f"{metrics['match_rate']}%" if "match_rate" in metrics else "—",
                   "from report parser")
    cols[3].metric("Images Analyzed",
                   int(metrics.get("total_images", 0)) if "total_images" in metrics else "—")
    cols[4].metric("Mean Confidence",
                   f"{metrics['mean_conf']:.3f}" if "mean_conf" in metrics else "—")

    # ── 1. AGGREGATE PLOTS ──────────────────────────────────────────────
    st.markdown("---")
    st.header("📈 Aggregate Analysis Plots")
    col1, col2 = st.columns(2)
    plots = scan["aggregate_pngs"]
    if not plots:
        st.info("No aggregate plots yet — run the evaluation pipeline.")
    else:
        for i, ax in enumerate([col1, col2]):
            if i < len(plots):
                with ax:
                    size_kb = plots[i].stat().st_size / 1024.0
                    st.image(str(plots[i]),
                             caption=f"{plots[i].name}  ·  {size_kb:.0f} KB",
                             use_container_width=True)

    # ── 2. TEXT REPORT (parsed highlights + raw) ────────────────────────
    st.markdown("---")
    st.header("📝 Evaluation Text Report")
    if scan["report_txt"] is None:
        st.warning("No text report found yet.")
    else:
        report_raw = scan["report_txt"].read_text(encoding='utf-8', errors='ignore')
        st.caption(f"Source: `{scan['report_txt'].name}`")
        with st.expander("🔎 View Raw Full Report", expanded=False):
            st.text(report_raw)

    # ── 3. GALLERY: graffiti_gallery_* ──────────────────────────────────
    render_grouped_section(
        title="Keeladi Graffiti Gallery (Top-3 Predictions)",
        icon="🖼️",
        grouped_paths=_group_pages(scan["gallery_pngs"]),
        description="Each sherd tile shows the graffiti photo + Top-3 Indus sign predictions with probability bars.",
        cols=1,
    )

    # ── 4. COMPARISON: graffiti_vs_indus_* ──────────────────────────────
    render_grouped_section(
        title="Graffiti ↔ Indus Sign (Side-by-Side Comparisons)",
        icon="⚖️",
        grouped_paths=_group_pages(scan["comparison_pngs"]),
        description="Left column = actual Keeladi/Brahmi photo · Right column = Top-3 predicted Indus sign IMAGES.",
        cols=1,
    )

    # ── 5. OTHER / MISC PNGs ────────────────────────────────────────────
    if scan["other_pngs"]:
        st.markdown("---")
        st.header("🗂️ Additional Generated Images")
        st.caption(f"{len(scan['other_pngs'])} file(s) not in known categories (auto-detected)")
        render_section_grid(scan["other_pngs"], cols=2)

    # ── 6. COMPLETE FILE INDEX with DOWNLOAD BUTTONS ────────────────────
    st.markdown("---")
    st.header("📂 Full Results Index (Download Any File)")
    st.caption("All PNG + text artifacts generated by the pipeline, click to download individually.")

    all_artifacts = []
    if scan["report_txt"] is not None:
        all_artifacts.append(scan["report_txt"])
    all_artifacts.extend(sorted(scan["all_pngs"], key=lambda x: x.name))

    if not all_artifacts:
        st.info("Nothing here yet.")
    else:
        rows = []
        for f in all_artifacts:
            try:
                sz = f.stat().st_size
                rows.append({
                    "File": f.name,
                    "Type": f.suffix.upper().lstrip('.'),
                    "Size KB": round(sz / 1024.0, 1),
                    "Path": str(f),
                })
            except Exception:
                pass
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"File": st.column_config.TextColumn(width="large"),
                                    "Path": st.column_config.TextColumn(width="medium")})

        st.markdown("#### Download any file:")
        pick = st.selectbox("Select file", [p.name for p in all_artifacts])
        pick_path = next(p for p in all_artifacts if p.name == pick)
        data_bytes = pick_path.read_bytes()
        st.download_button(
            label=f"⬇️  Download {pick_path.name}",
            data=data_bytes,
            file_name=pick_path.name,
            mime="image/png" if pick_path.suffix.lower() == '.png' else "text/plain",
        )

    # ── 7. Training Results (kept from original dashboard) ──────────────
    st.markdown("---")
    st.header("🎯 Training Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Information")
        if training["model_exists"]:
            st.success("✅ Model trained successfully")
            st.info("📁 Model saved as: `indus_classifier.keras`")
            st.info(f"📊 Total Indus classes: {len(training['class_names'])}")
        else:
            st.error("❌ Model not found")
    with col2:
        st.subheader("Training Progress")
        st.info("🔄 **Training completed (demo run)**")
        st.info("📈 **Best accuracy achieved: demo**")
        st.info("🎛️ **Early stopping enabled**")
        st.info("⏱️  Re-run training for real metrics.")

    # ── 8. Indus Sign Classes ───────────────────────────────────────────
    st.markdown("---")
    st.header("🔤 Indus Sign Classes")
    if training["class_names"]:
        st.info(f"Total of {len(training['class_names'])} primary core signs:")
        cols = st.columns(5)
        for idx, class_name in enumerate(training["class_names"][:20]):
            with cols[idx % 5]:
                st.text(class_name)
        with st.expander(f"View All {len(training['class_names'])} Classes"):
            for class_name in training["class_names"]:
                st.text(f"• {class_name}")

    # ── 9. Model Architecture ───────────────────────────────────────────
    st.markdown("---")
    st.header("🧠 Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CNN Architecture")
        st.markdown("""
        **Input Layer**: 64x64x1 (grayscale images)
        **Convolutional Blocks**: 32→64→128 filters, 3x3 kernels
        **Classification Head**: Dense 256→128→N classes (Softmax)
        **Regularization**: BatchNorm + Dropout
        """)
    with col2:
        st.subheader("Training Configuration")
        st.markdown("""
        **Optimizer**: Adam (lr=0.001)
        **Loss**: Sparse Categorical Crossentropy
        **Callbacks**: Early Stopping (patience=10), Reduce LR on Plateau
        **Batch Size**: 16/32 · **Val Split**: 20%
        """)

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🏺 Indus-Keeladi CNN Project  ·  Civilization Link Analysis through Deep Learning  ·  "
        "auto-renders from `models/evaluation_results/`"
    )


if __name__ == "__main__":
    main()
