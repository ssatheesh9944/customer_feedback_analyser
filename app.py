import streamlit as st
from transformers import pipeline
import pandas as pd


@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error("Failed to load the Hugging Face model. Make sure `transformers` is installed and you have internet access.")
        raise e


def main():
    st.title("AI Customer Feedback Analyzer")
    st.write("Enter a customer review below and click **Analyze** to get a sentiment prediction with a confidence score.")

    review = st.text_area("Customer review", height=160)

    if st.button("Analyze"):
        if not review or not review.strip():
            st.warning("Please enter a customer review to analyze.")
            return

        with st.spinner("Analyzing review..."):
            nlp = load_sentiment_pipeline()
            result = nlp(review[:1000])[0]

        label = result.get("label", "N/A")
        score = result.get("score", 0.0)

        if label.lower().startswith("pos"):
            st.success(f"Sentiment: {label} \U0001F600")
        elif label.lower().startswith("neg"):
            st.error(f"Sentiment: {label} \U0001F61E")
        else:
            st.info(f"Sentiment: {label}")

        st.write(f"**Confidence:** {score:.1%}")

    st.markdown("---")
    st.subheader("Analyze multiple reviews from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file (must contain a column named 'review')", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("Could not read the uploaded CSV. Make sure it's a valid CSV file.")
            return

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'.")
        else:
            # Normalize and ignore empty reviews
            df["review"] = df["review"].astype(str).str.strip()
            df = df[df["review"].notna() & (df["review"] != "")].copy()

            if df.empty:
                st.warning("No non-empty reviews found in the uploaded CSV.")
            else:
                with st.spinner("Analyzing reviews..."):
                    nlp = load_sentiment_pipeline()
                    texts = [t[:1000] for t in df["review"].tolist()]
                    results = nlp(texts, truncation=True)

                labels = [r.get("label", "N/A") for r in results]
                scores = [r.get("score", 0.0) for r in results]

                df["sentiment"] = labels
                df["confidence"] = scores

                st.write("### Results")
                st.dataframe(df)

                total = len(df)
                positive = sum(1 for l in labels if str(l).lower().startswith("pos"))
                negative = sum(1 for l in labels if str(l).lower().startswith("neg"))

                st.write(f"**Total processed:** {total}")
                st.write(f"**Positive:** {positive}")
                st.write(f"**Negative:** {negative}")


if __name__ == "__main__":
    main()

