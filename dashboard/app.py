"""Local, supervised research dashboard for the IDRiD demo candidate."""

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model import DEFAULT_CHECKPOINT, check_fundus_appearance, load_model, predict_image


st.set_page_config(
    page_title="IDRiD DR Research Demo",
    page_icon=":material/visibility:",
    layout="centered",
)


@st.cache_resource
def get_model():
    return load_model()


st.title("Retinal image research demo")
st.caption("IDRiD EfficientNet-B0 candidate | 300 x 300 input")
st.warning(
    "Research and education only. This model is not clinically validated and "
    "must not be used to diagnose, treat, or make care decisions. A qualified "
    "eye-care professional must review every image."
)
st.info(
    "This is a binary model: Normal (grade 0) or possible DR (grades 1-4). "
    "It does not predict a DR grade. The 0.50 cutoff is a demonstration setting, "
    "not a medically approved threshold."
)

if not DEFAULT_CHECKPOINT.exists():
    st.error(
        "The model file is missing. Place "
        "models/IDRiD_demo_candidate_300px.pt in the project folder."
    )
    st.stop()

uploaded = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"],
    help="The image is processed in memory by this app and is not saved to a file by the app.",
)

if uploaded is not None:
    if uploaded.size > 20 * 1024 * 1024:
        st.error("This file is over the 20 MB demo limit. Choose a smaller image.")
        st.stop()

    try:
        image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The selected file could not be opened as an image.")
        st.stop()

    looks_like_fundus, orange_fraction = check_fundus_appearance(image)

    col_image, col_result = st.columns([1.15, 1])
    with col_image:
        st.image(image, caption=f"Uploaded image - {image.width} x {image.height}px", use_column_width=True)
    with col_result:
        st.subheader("Model output")
        if not looks_like_fundus:
            st.error(
                "Image not accepted: this does not look like an IDRiD-style retinal photo. "
                "No DR prediction was run. Please upload a retinal fundus image."
            )
            st.caption(
                "Automatic check: only about "
                f"{orange_fraction:.0%} of sampled pixels had the orange-red appearance "
                "seen in this project’s retinal photos. This simple check can make mistakes."
            )
            st.stop()
        st.caption("Retinal-photo appearance check passed. This is only a basic input check.")
        if image.width < 300 or image.height < 300:
            st.warning("This image is small. Results on small images have not been validated.")

        if st.button("Run demo prediction", type="primary"):
            try:
                with st.spinner("Loading the model and analyzing the image..."):
                    model, device = get_model()
                    dr_score = predict_image(model, device, image)
                predicted_class = 1 if dr_score >= 0.5 else 0
                display_name = "Possible DR" if predicted_class == 1 else "Model class: Normal"
                st.metric("Model output", display_name)
                st.metric("DR score (uncalibrated)", f"{dr_score:.1%}")
                st.progress(dr_score)
                st.caption(f"Running on: {device}")
                st.warning(
                    "This score is not calibrated for clinical risk. It is a model "
                    "output, not a diagnosis. Image quality, camera, and patient "
                    "population can change performance."
                )
            except Exception as error:
                st.error(f"Could not run the model: {error}")

st.divider()
st.caption(
    "Local prototype. Do not upload identifiable patient images to a hosted service "
    "until privacy, security, access, and retention have been reviewed."
)
