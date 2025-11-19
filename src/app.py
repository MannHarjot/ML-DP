import gc
import streamlit as st

from dp_pipeline import (
    load_dataframe,
    validate_dataframe,
    detect_quasi_identifiers,
    apply_strong_dp,
    dataframe_to_csv_bytes,
)
from backend_model import train_global_model


def main():
    st.title("🛡️ ML + DP Anonymization Service (v3)")
    st.write(
        """
        Upload a dataset, choose a privacy level (epsilon), and download a **differentially private**
        version of your data.

        **Privacy note:**  
        Your raw data is only kept in memory during this session.  
        It is not stored on disk. Only the global model parameters are saved.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"]
    )

    epsilon = st.slider(
        "Privacy parameter ε (smaller = more privacy, less accuracy)",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.01,
    )

    if uploaded_file is not None:
        st.subheader("Step 1: Load & Validate Dataset")

        try:
            df = load_dataframe(uploaded_file, uploaded_file.name)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return

        st.write("### 🔍 Preview (first 5 rows):")
        st.dataframe(df.head())

        _, warnings = validate_dataframe(df)
        if warnings:
            st.warning("⚠️ Warnings:\n- " + "\n- ".join(warnings))

        qids = detect_quasi_identifiers(df)
        if qids:
            st.info(f"🔐 Possible quasi-identifiers detected: **{', '.join(qids)}**")
        else:
            st.info("No obvious quasi-identifiers detected.")

        # Step 2: Backend training (efficient)
        st.subheader("Step 2: Backend Model Training (Invisible to User)")
        with st.spinner("Training / updating backend model on 80% of your raw data..."):
            try:
                acc = train_global_model(df)
                if acc is not None:
                    st.success(f"Backend model updated. Validation accuracy: **{acc:.4f}**")
                else:
                    st.info(
                        "Dataset not suitable for backend ML training (e.g., no clear label). "
                        "DP processing will still continue."
                    )
            except Exception as e:
                st.warning(f"Backend model training skipped due to error: {e}")

        # Step 3: Apply DP (vectorized, strong)
        st.subheader("Step 3: Apply Differential Privacy")
        with st.spinner("Applying strong DP transform to your entire dataset..."):
            try:
                dp_df = apply_strong_dp(df, epsilon=epsilon)
            except Exception as e:
                st.error(f"❌ Error during DP transformation: {e}")
                return

        st.write("### 🔐 Preview of DP Dataset (first 5 rows):")
        st.dataframe(dp_df.head())

        # Memory cleanup: delete raw df
        del df
        gc.collect()

        # Step 4: Download DP dataset
        st.subheader("Step 4: Download")
        csv_bytes = dataframe_to_csv_bytes(dp_df)
        st.download_button(
            label="⬇️ Download DP Dataset (CSV)",
            data=csv_bytes,
            file_name=f"dp_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
        )

        # Cleanup DP df too
        del dp_df
        gc.collect()


if __name__ == "__main__":
    main()
