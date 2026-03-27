import streamlit as st
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return f"Could not read: {path}"


st.set_page_config(page_title="Genome AI Agents - Dashboard", layout="wide")
st.title("Genome AI Agents - Pipeline Dashboard")

reports_dir = Path("reports")
final_report_path = reports_dir / "final_report.md"
quality_report_path = reports_dir / "quality_report.md"
annotation_report_path = reports_dir / "annotation_report.md"
al_report_path = reports_dir / "al_report.md"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Final Report")
    if final_report_path.exists():
        st.markdown(read_text(final_report_path))
    else:
        st.write("Run `python run_pipeline.py` to generate reports.")

with col2:
    st.subheader("Quality / Annotation / Active Learning")
    if quality_report_path.exists():
        st.markdown(read_text(quality_report_path))
    if annotation_report_path.exists():
        st.markdown(read_text(annotation_report_path))
    if al_report_path.exists():
        st.markdown(read_text(al_report_path))

lc_img = reports_dir / "learning_curve.png"
if lc_img.exists():
    st.divider()
    st.subheader("Active Learning Learning Curve")
    st.image(str(lc_img))

