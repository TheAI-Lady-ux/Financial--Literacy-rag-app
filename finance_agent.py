
"""
finance_agent.py

Combines:
1. Spending-data analysis
2. RAG financial-literacy guidance
3. A personalized recommendation
"""

from src. spending_analyzer import analyze_spending, money
from src.rag_assistant import ask_rag


def build_finance_question(results: dict) -> str:
    """Create a question for the RAG system using spending results."""

    summary = results["summary"]
    needs_wants = results["needs_wants"]
    impulse = results["impulse"]
    savings = results["savings"]

    question = f"""
A spending dataset contains {summary['transaction_count']} transactions.

Total spending: {money(summary['total_spending'])}
Average transaction: {money(summary['average_transaction'])}

Need spending: {money(needs_wants.get('need_spending', 0))}
Want spending: {money(needs_wants.get('want_spending', 0))}
Want-spending percentage: {needs_wants.get('want_percentage', 0):.2f}%

Potential impulse transactions: {impulse.get('impulse_count', 0)}
Potential impulse spending: {money(impulse.get('total_impulse_spending', 0))}
Impulse-spending percentage:
{impulse.get('impulse_spending_percentage', 0):.2f}%

Estimated savings opportunity:
{money(savings.get('estimated_savings_opportunity', 0))}

Using the uploaded financial-literacy documents, explain this spending
pattern and provide five practical steps for reducing impulse spending.
"""

    return question.strip()


def run_finance_agent() -> dict:
    """Run the spending analyzer and RAG system."""

    print("\nAnalyzing spending data...")

    spending_results = analyze_spending()

    print("Spending analysis completed.")
    print("Retrieving financial-literacy guidance...")

    finance_question = build_finance_question(
        spending_results
    )

    rag_result = ask_rag(
        question=finance_question,
        top_k=4,
    )

    return {
        "spending_results": spending_results,
        "question": finance_question,
        "recommendation": rag_result["answer"],
        "sources": rag_result["sources"],
    }


def display_agent_result(result: dict) -> None:
    """Display the recommendation and document sources."""

    print("\n" + "=" * 70)
    print("AI PERSONAL FINANCE COACH")
    print("=" * 70)

    print("\nPERSONALIZED RECOMMENDATION")
    print("-" * 70)
    print(result["recommendation"])

    print("\nDOCUMENT SOURCES")
    print("-" * 70)

    sources = result.get("sources", [])

    if not sources:
        print("No document sources were retrieved.")

    for number, source in enumerate(
        sources,
        start=1,
    ):
        print(
            f"{number}. {source.get('source', 'Unknown document')} "
            f"- Page {source.get('page_number', 'Unknown')}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        agent_result = run_finance_agent()
        display_agent_result(agent_result)

    except Exception as error:
        print("\nThe finance agent could not run.")
        print(f"\nError: {error}")

        print(
            "\nCheck that:"
            "\n1. spending_analyzer.py works."
            "\n2. rag_pipeline.py works."
            "\n3. Ollama is running."
            "\n4. llama3.2 is installed."
            "\n5. The Chroma vector database exists."
        )

def create_finance_question(question: str) -> str:
    """Create a prompt for the financial AI assistant."""

    return f"""
You are a personal finance assistant.

Answer the following question clearly and practically:

{question}

Provide educational guidance only. Do not present the response as
professional financial advice.
"""