import gc
import streamlit as st
import pandas as pd

from dp_pipeline import (
    load_dataframe,
    validate_dataframe,
    detect_quasi_identifiers,
    apply_dp,
    dataframe_to_csv_bytes,
)

from backend_model import train_global_model


def main():
    st.title("🛡️ ML + DP Anonymization Service")
    st.write(
        """
        Upload a dataset, choose a privacy level (epsilon), and download a **differentially private** 
        version of your data.

        ⚠️ *Your raw data is only stored temporarily in RAM during processing. 
        It is never saved on disk and is deleted immediately after processing.*
        """
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)", 
        type=["csv", "xlsx", "xls"]
    )

    epsilon = st.slider(
        "Privacy parameter ε (smaller = more privacy, less accuracy)",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.01,
    )

    if uploaded_file is not None:
        st.subheader("Step 1: Loading & validating dataset")

        try:
            df = load_dataframe(uploaded_file, uploaded_file.name)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return

        st.write("### 🔍 Preview of your dataset (first 5 rows):")
        st.dataframe(df.head())

        # Validate dataset
        is_valid, warnings = validate_dataframe(df)
        if warnings:
            st.warning("⚠️ Dataset Warnings:\n- " + "\n- ".join(warnings))

        # Identify quasi-identifiers
        qids = detect_quasi_identifiers(df)
        if qids:
            st.info(f"🔐 Detected possible Quasi-Identifiers: **{', '.join(qids)}**")
        else:
            st.info("No obvious quasi-identifiers found.")

        # Backend model training
        st.subheader("Step 2: Backend Model Training (Invisible to User)")
        with st.spinner("Training backend model on raw data (80/20 split)..."):
            try:
                acc = train_global_model(df)
                if acc is not None:
                    st.success(f"Backend model updated ✔️ Validation accuracy: **{acc:.4f}**")
                else:
                    st.info(
                        """
                        ℹ️ Dataset not suitable for backend model training 
                        (e.g., not enough numeric or categorical structure).  
                        DP processing will continue normally.
                        """
                    )
            except Exception as e:
                st.warning(f"⚠️ Backend model training skipped due to error: {e}")

        # Apply strong DP
        st.subheader("Step 3: Applying Differential Privacy")

        with st.spinner("Applying strong differential privacy protections..."):
            try:
                dp_df = apply_dp(df, epsilon=epsilon)
            except Exception as e:
                st.error(f"❌ Error during DP transformation: {e}")
                return

        st.write("### 🔐 Preview of Differentally Private Dataset:")
        st.dataframe(dp_df.head())

        # Delete raw data for privacy
        del df
        gc.collect()

        # Download button
        st.subheader("Step 4: Download Your Differentially Private Dataset")
        csv_bytes = dataframe_to_csv_bytes(dp_df)

        st.download_button(
            label="⬇️ Download DP Dataset (CSV)",
            data=csv_bytes,
            file_name=f"dp_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
        )

        # Cleanup DP df
        del dp_df
        gc.collect()


if __name__ == "__main__":
    main()
