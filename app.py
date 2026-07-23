"""
app.py

Streamlit application for the
AI-Powered Personal Finance Coach.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.finance_agent import create_finance_question
from src.rag_assistant import ask_rag
from src.spending_analyzer import analyze_spending, money


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Personal Finance Coach",
    page_icon="💰",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_uploaded_file(uploaded_file) -> Path:
    """Save an uploaded Excel file temporarily."""

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        return Path(temporary_file.name)


def display_sources(sources: list[dict]) -> None:
    """Display the sources returned by the RAG assistant."""

    if not sources:
        st.info("No document sources were retrieved.")
        return

    for number, source in enumerate(sources, start=1):
        document_name = source.get(
            "source",
            "Unknown document",
        )

        page_number = source.get(
            "page_number",
            "Unknown",
        )

        st.write(
            f"{number}. {document_name} — Page {page_number}"
        )


# ---------------------------------------------------------
# App heading
# ---------------------------------------------------------

st.title("💰 AI-Powered Personal Finance Coach")

st.write(
    """
    This application analyzes spending transactions, identifies
    potential impulse purchases and retrieves financial-literacy
    guidance from stored documents.
    """
)

st.info(
    "This application is for educational purposes only."
)

# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------

analysis_tab, rag_tab, recommendation_tab = st.tabs(
    [
        "Spending Analysis",
        "RAG Assistant",
        "AI Recommendation",
    ]
)

# ---------------------------------------------------------
# Tab 1: Spending analysis
# ---------------------------------------------------------

with analysis_tab:
    st.header("Spending Analysis")

    uploaded_file = st.file_uploader(
        "Upload an Excel spending dataset",
        type=["xlsx", "xls"],
    )

    use_default_dataset = st.checkbox(
        "Use the default 300-transaction dataset",
        value=True,
    )

    if st.button(
        "Analyze Spending",
        type="primary",
    ):
        try:
            with st.spinner("Analyzing spending data..."):
                if uploaded_file is not None:
                    file_path = save_uploaded_file(
                        uploaded_file
                    )

                    results = analyze_spending(
                        file_path=file_path
                    )

                elif use_default_dataset:
                    results = analyze_spending()

                else:
                    st.warning(
                        "Upload an Excel file or select "
                        "the default dataset."
                    )
                    st.stop()

                st.session_state[
                    "spending_results"
                ] = results

            st.success(
                "Spending analysis completed."
            )

        except Exception as error:
            st.error(
                f"Spending analysis error: {error}"
            )

    if "spending_results" in st.session_state:
        results = st.session_state[
            "spending_results"
        ]

        summary = results["summary"]
        needs_wants = results["needs_wants"]
        impulse = results["impulse"]
        savings = results["savings"]

        column1, column2, column3, column4 = st.columns(4)

        column1.metric(
            "Total Spending",
            money(summary["total_spending"]),
        )

        column2.metric(
            "Transactions",
            summary["transaction_count"],
        )

        column3.metric(
            "Impulse Spending",
            money(
                impulse.get(
                    "total_impulse_spending",
                    0,
                )
            ),
        )

        column4.metric(
            "Savings Opportunity",
            money(
                savings.get(
                    "estimated_savings_opportunity",
                    0,
                )
            ),
        )

        st.subheader("Spending by Category")

        category_table = results["categories"]

        st.dataframe(
            category_table,
            use_container_width=True,
            hide_index=True,
        )

        if not category_table.empty:
            category_column = category_table.columns[0]

            chart_data = (
                category_table[
                    [
                        category_column,
                        "Total_Spending",
                    ]
                ]
                .set_index(category_column)
            )

            st.bar_chart(chart_data)

        st.subheader("Needs Versus Wants")

        if needs_wants.get("available"):
            st.dataframe(
                needs_wants["table"],
                use_container_width=True,
                hide_index=True,
            )

            need_column, want_column = st.columns(2)

            need_column.metric(
                "Need Spending",
                money(
                    needs_wants.get(
                        "need_spending",
                        0,
                    )
                ),
            )

            want_column.metric(
                "Want Spending",
                money(
                    needs_wants.get(
                        "want_spending",
                        0,
                    )
                ),
            )

        else:
            st.warning(
                needs_wants.get(
                    "message",
                    "Needs-versus-wants data is unavailable.",
                )
            )

        st.subheader("Potential Impulse Spending")

        if impulse.get("available"):
            impulse_column1, impulse_column2, impulse_column3 = (
                st.columns(3)
            )

            impulse_column1.metric(
                "Impulse Transactions",
                impulse.get(
                    "impulse_count",
                    0,
                ),
            )

            impulse_column2.metric(
                "Impulse Spending",
                money(
                    impulse.get(
                        "total_impulse_spending",
                        0,
                    )
                ),
            )

            impulse_column3.metric(
                "Impulse-Spending Percentage",
                (
                    f"{impulse.get('impulse_spending_percentage', 0):.2f}%"
                ),
            )

        else:
            st.warning(
                impulse.get(
                    "message",
                    "Impulse-spending data is unavailable.",
                )
            )

        st.subheader("Monthly Spending")

        monthly_table = results["monthly"]

        if monthly_table.empty:
            st.info(
                "Monthly spending information is unavailable."
            )

        else:
            st.dataframe(
                monthly_table,
                use_container_width=True,
                hide_index=True,
            )

            month_column = monthly_table.columns[0]

            monthly_chart = (
                monthly_table[
                    [
                        month_column,
                        "Total_Spending",
                    ]
                ]
                .set_index(month_column)
            )

            st.line_chart(monthly_chart)

        st.subheader("Key Insights")

        for insight in results["insights"]:
            st.write(f"• {insight}")


# ---------------------------------------------------------
# Tab 2: RAG assistant
# ---------------------------------------------------------

with rag_tab:
    st.header("Financial Literacy RAG Assistant")

    question = st.text_area(
        "Ask a financial-literacy question",
        placeholder=(
            "How can a spending plan help reduce "
            "impulse purchases?"
        ),
        height=120,
    )

    if st.button(
        "Ask RAG Assistant",
        type="primary",
    ):
        if not question.strip():
            st.warning(
                "Please enter a question."
            )

        else:
            try:
                with st.spinner(
                    "Retrieving financial-literacy guidance..."
                ):
                    result = ask_rag(
                        question=question,
                        top_k=4,
                    )

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                st.subheader("Sources")

                display_sources(
                    result["sources"]
                )

            except Exception as error:
                st.error(
                    f"RAG assistant error: {error}"
                )

                st.info(
                    "Confirm that Ollama is running, "
                    "llama3.2 is installed and the "
                    "vector database has been created."
                )


# ---------------------------------------------------------
# Tab 3: AI recommendation
# ---------------------------------------------------------

with recommendation_tab:
    st.header("Personalized AI Recommendation")

    if "spending_results" not in st.session_state:
        st.warning(
            "Analyze the spending dataset first."
        )

    else:
        results = st.session_state[
            "spending_results"
        ]

        if st.button(
            "Generate Recommendation",
            type="primary",
        ):
            try:
                finance_question = create_finance_question(
                    results
                )

                with st.spinner(
                    "Generating a personalized recommendation..."
                ):
                    recommendation = ask_rag(
                        question=finance_question,
                        top_k=4,
                    )

                st.session_state[
                    "recommendation"
                ] = recommendation

            except Exception as error:
                st.error(
                    f"Recommendation error: {error}"
                )

        if "recommendation" in st.session_state:
            recommendation = st.session_state[
                "recommendation"
            ]

            st.subheader("Recommendation")

            st.write(
                recommendation["answer"]
            )

            st.subheader("Sources")

            display_sources(
                recommendation["sources"]
            )