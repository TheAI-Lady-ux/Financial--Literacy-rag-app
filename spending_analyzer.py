"""
spending_analyzer.py

AI-Powered Personal Finance Coach

This module reads the spending transaction Excel dataset and analyzes:

1. Total spending
2. Needs versus wants
3. Spending by category
4. Potential impulse purchases
5. Monthly spending patterns
6. Payment-method patterns
7. Budgeted versus unbudgeted spending
8. Income and spending ratios
9. Savings opportunities
10. Personalized spending insights

Run this file from the main project folder with:

    python src/spending_analyzer.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"

DEFAULT_DATA_FILE = (
    DATA_DIRECTORY
    / "Finance_AI_Spending_Dataset_300_Transactions.xlsx"
)

# ---------------------------------------------------------
# Accepted dataset column names
# ---------------------------------------------------------

DATE_COLUMN_OPTIONS = [
    "Date",
    "Transaction_Date",
    "Transaction Date",
]

AMOUNT_COLUMN_OPTIONS = [
    "Amount_USD",
    "Amount",
    "Transaction_Amount",
    "Transaction Amount",
    "Purchase_Amount",
    "Purchase Amount",
]

INCOME_COLUMN_OPTIONS = [
    "Monthly_Income_USD",
    "Monthly Income USD",
    "Monthly_Income",
    "Monthly Income",
    "Income_USD",
    "Income",
]

CATEGORY_COLUMN_OPTIONS = [
    "Category",
    "Spending_Category",
    "Spending Category",
]

NEED_WANT_COLUMN_OPTIONS = [
    "Need_or_Want",
    "Need or Want",
    "Need/Want",
    "Need_Want",
    "Type",
]

IMPULSE_COLUMN_OPTIONS = [
    "Potential_Impulse",
    "Potential Impulse",
    "Impulse_Purchase",
    "Impulse Purchase",
    "Impulse",
]

PAYMENT_METHOD_COLUMN_OPTIONS = [
    "Payment_Method",
    "Payment Method",
    "Payment_Type",
    "Payment Type",
]

BUDGETED_COLUMN_OPTIONS = [
    "Budgeted",
    "Is_Budgeted",
    "Is Budgeted",
]

DESCRIPTION_COLUMN_OPTIONS = [
    "Description",
    "Merchant",
    "Transaction_Description",
    "Transaction Description",
]

MONTH_COLUMN_OPTIONS = [
    "Month",
    "Transaction_Month",
    "Transaction Month",
]
# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def normalize_column_name(value: Any) -> str:
    """Normalize a column name for reliable matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def find_column(
    dataframe: pd.DataFrame,
    options: list[str],
    required: bool = True,
) -> str | None:
    """
    Find a column using several possible names.

    Matching ignores capitalization and treats spaces, hyphens,
    underscores, and slashes consistently.
    """
    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for option in options:
        match = normalized_columns.get(
            normalize_column_name(option)
        )
        if match is not None:
            return str(match)

    if required:
        raise ValueError(
            "A required column was not found.\n"
            f"Expected one of: {options}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None

def normalize_yes_no(value: Any) -> str:
    """Convert common yes/no formats into Yes or No."""
    if pd.isna(value):
        return "No"

    normalized = str(value).strip().lower()

    if normalized in {
        "yes",
        "y",
        "true",
        "1",
        "impulse",
        "potential impulse",
    }:
        return "Yes"

    return "No"


def normalize_need_want(value: Any) -> str:
    """Standardize need/want values."""
    if pd.isna(value):
        return "Unknown"

    normalized = str(value).strip().lower()

    if normalized in {
        "need",
        "needs",
        "necessary",
        "essential",
    }:
        return "Need"

    if normalized in {
        "want",
        "wants",
        "nonessential",
        "non-essential",
        "discretionary",
    }:
        return "Want"

    return "Unknown"


def money(value: float) -> str:
    """Format a number as US currency."""
    return f"${value:,.2f}"


# ---------------------------------------------------------
# Load and clean the Excel dataset
# ---------------------------------------------------------

def load_spending_data(
    file_path: str | Path = DEFAULT_DATA_FILE,
) -> pd.DataFrame:
    """Load and clean the spending transaction dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            "The spending dataset was not found at:\n"
            f"{file_path.resolve()}\n\n"
            "Place the Excel file inside the project's data folder."
        )

    dataframe = pd.read_excel(
        file_path,
        sheet_name="Transactions",
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError("The spending dataset is empty.")

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    # Remove accidental exported index columns.
    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.str.match(
            r"^Unnamed:",
            case=False,
        ),
    ]

    dataframe = dataframe.dropna(
        how="all"
    ).copy()

    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    dataframe[amount_column] = pd.to_numeric(
        dataframe[amount_column],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[amount_column]
    ).copy()

    # Negative values in this dataset represent savings transfers,
    # not expenses. Keep them in the source workbook but exclude them
    # from spending calculations.
    dataframe = dataframe[
        dataframe[amount_column] >= 0
    ].copy()

    date_column = find_column(
        dataframe,
        DATE_COLUMN_OPTIONS,
        required=False,
    )

    if date_column:
        dataframe[date_column] = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
        )

    income_column = find_column(
        dataframe,
        INCOME_COLUMN_OPTIONS,
        required=False,
    )

    if income_column:
        dataframe[income_column] = pd.to_numeric(
            dataframe[income_column],
            errors="coerce",
        )

    need_want_column = find_column(
        dataframe,
        NEED_WANT_COLUMN_OPTIONS,
        required=False,
    )

    if need_want_column:
        dataframe[need_want_column] = (
            dataframe[need_want_column]
            .apply(normalize_need_want)
        )

    impulse_column = find_column(
        dataframe,
        IMPULSE_COLUMN_OPTIONS,
        required=False,
    )

    if impulse_column:
        dataframe[impulse_column] = (
            dataframe[impulse_column]
            .apply(normalize_yes_no)
        )

    budgeted_column = find_column(
        dataframe,
        BUDGETED_COLUMN_OPTIONS,
        required=False,
    )

    if budgeted_column:
        dataframe[budgeted_column] = (
            dataframe[budgeted_column]
            .apply(normalize_yes_no)
        )

    return dataframe


# ---------------------------------------------------------
# Basic spending summary
# ---------------------------------------------------------

def calculate_basic_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate general transaction statistics."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    return {
        "transaction_count": int(len(dataframe)),
        "total_spending": float(
            dataframe[amount_column].sum()
        ),
        "average_transaction": float(
            dataframe[amount_column].mean()
        ),
        "median_transaction": float(
            dataframe[amount_column].median()
        ),
        "highest_transaction": float(
            dataframe[amount_column].max()
        ),
        "lowest_transaction": float(
            dataframe[amount_column].min()
        ),
    }


# ---------------------------------------------------------
# Category analysis
# ---------------------------------------------------------

def analyze_categories(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate spending and transaction counts by category."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    category_column = find_column(
        dataframe,
        CATEGORY_COLUMN_OPTIONS,
    )

    result = (
        dataframe.groupby(
            category_column,
            dropna=False,
        )
        .agg(
            Total_Spending=(
                amount_column,
                "sum",
            ),
            Transaction_Count=(
                amount_column,
                "count",
            ),
            Average_Transaction=(
                amount_column,
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Total_Spending",
            ascending=False,
        )
    )

    total_spending = float(
        result["Total_Spending"].sum()
    )

    result["Percentage_of_Total"] = (
        result["Total_Spending"]
        / total_spending
        * 100
        if total_spending > 0
        else 0.0
    )

    for column in [
        "Total_Spending",
        "Average_Transaction",
        "Percentage_of_Total",
    ]:
        result[column] = result[column].round(2)

    return result

# ---------------------------------------------------------
# Needs versus wants
# ---------------------------------------------------------

def analyze_needs_and_wants(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Compare essential and discretionary spending."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    need_want_column = find_column(
        dataframe,
        NEED_WANT_COLUMN_OPTIONS,
        required=False,
    )

    if not need_want_column:
        return {
            "available": False,
            "message": (
                "The dataset does not contain "
                "a Need_or_Want column."
            ),
            "table": pd.DataFrame(),
        }

    grouped = (
        dataframe.groupby(
            need_want_column,
            dropna=False,
        )
        .agg(
            Total_Spending=(
                amount_column,
                "sum",
            ),
            Transaction_Count=(
                amount_column,
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "Total_Spending",
            ascending=False,
        )
    )

    total_spending = float(
        dataframe[amount_column].sum()
    )

    grouped["Percentage"] = (
        grouped["Total_Spending"]
        / total_spending
        * 100
        if total_spending > 0
        else 0.0
    )

    grouped["Total_Spending"] = (
        grouped["Total_Spending"].round(2)
    )
    grouped["Percentage"] = (
        grouped["Percentage"].round(2)
    )

    need_spending = float(
        dataframe.loc[
            dataframe[need_want_column] == "Need",
            amount_column,
        ].sum()
    )

    want_spending = float(
        dataframe.loc[
            dataframe[need_want_column] == "Want",
            amount_column,
        ].sum()
    )

    want_percentage = (
        want_spending / total_spending * 100
        if total_spending > 0
        else 0.0
    )

    return {
        "available": True,
        "need_spending": need_spending,
        "want_spending": want_spending,
        "want_percentage": want_percentage,
        "table": grouped,
    }


# ---------------------------------------------------------
# Impulse-spending analysis
# ---------------------------------------------------------

def analyze_impulse_spending(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze transactions marked as potential impulses."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    impulse_column = find_column(
        dataframe,
        IMPULSE_COLUMN_OPTIONS,
        required=False,
    )

    if not impulse_column:
        return {
            "available": False,
            "message": (
                "The dataset does not contain "
                "a Potential_Impulse column."
            ),
            "transactions": pd.DataFrame(),
            "by_category": pd.DataFrame(),
        }

    impulse_transactions = dataframe[
        dataframe[impulse_column] == "Yes"
    ].copy()

    total_impulse_spending = float(
        impulse_transactions[amount_column].sum()
    )
    impulse_count = int(
        len(impulse_transactions)
    )
    total_spending = float(
        dataframe[amount_column].sum()
    )
    total_transactions = int(
        len(dataframe)
    )

    impulse_spending_percentage = (
        total_impulse_spending
        / total_spending
        * 100
        if total_spending > 0
        else 0.0
    )

    impulse_transaction_percentage = (
        impulse_count
        / total_transactions
        * 100
        if total_transactions > 0
        else 0.0
    )

    category_column = find_column(
        dataframe,
        CATEGORY_COLUMN_OPTIONS,
        required=False,
    )

    if category_column and not impulse_transactions.empty:
        impulse_by_category = (
            impulse_transactions.groupby(
                category_column
            )[amount_column]
            .agg(
                ["sum", "count", "mean"]
            )
            .reset_index()
            .rename(
                columns={
                    "sum": "Impulse_Spending",
                    "count": "Impulse_Count",
                    "mean": "Average_Impulse_Amount",
                }
            )
            .sort_values(
                "Impulse_Spending",
                ascending=False,
            )
        )

        impulse_by_category[
            "Impulse_Spending"
        ] = impulse_by_category[
            "Impulse_Spending"
        ].round(2)

        impulse_by_category[
            "Average_Impulse_Amount"
        ] = impulse_by_category[
            "Average_Impulse_Amount"
        ].round(2)
    else:
        impulse_by_category = pd.DataFrame()

    return {
        "available": True,
        "impulse_count": impulse_count,
        "total_impulse_spending": (
            total_impulse_spending
        ),
        "impulse_spending_percentage": (
            impulse_spending_percentage
        ),
        "impulse_transaction_percentage": (
            impulse_transaction_percentage
        ),
        "transactions": impulse_transactions,
        "by_category": impulse_by_category,
    }


# ---------------------------------------------------------
# Monthly spending
# ---------------------------------------------------------

def analyze_monthly_spending(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate spending totals by month."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    date_column = find_column(
        dataframe,
        DATE_COLUMN_OPTIONS,
        required=False,
    )

    month_column = find_column(
        dataframe,
        MONTH_COLUMN_OPTIONS,
        required=False,
    )

    working_data = dataframe.copy()

    if date_column:
        valid_dates = working_data.dropna(
            subset=[date_column]
        ).copy()

        if not valid_dates.empty:
            valid_dates["Analysis_Month"] = (
                valid_dates[date_column]
                .dt.to_period("M")
                .astype(str)
            )

            return (
                valid_dates.groupby(
                    "Analysis_Month"
                )
                .agg(
                    Total_Spending=(
                        amount_column,
                        "sum",
                    ),
                    Transaction_Count=(
                        amount_column,
                        "count",
                    ),
                    Average_Transaction=(
                        amount_column,
                        "mean",
                    ),
                )
                .reset_index()
                .sort_values(
                    "Analysis_Month"
                )
                .round(2)
            )

    if month_column:
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        working_data[month_column] = pd.Categorical(
            working_data[month_column],
            categories=month_order,
            ordered=True,
        )

        return (
            working_data.groupby(
                month_column,
                observed=False,
            )
            .agg(
                Total_Spending=(
                    amount_column,
                    "sum",
                ),
                Transaction_Count=(
                    amount_column,
                    "count",
                ),
                Average_Transaction=(
                    amount_column,
                    "mean",
                ),
            )
            .reset_index()
            .dropna(
                subset=[month_column]
            )
            .round(2)
        )

    return pd.DataFrame()

# ---------------------------------------------------------
# Payment-method analysis
# ---------------------------------------------------------

def analyze_payment_methods(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze spending by payment method."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    payment_column = find_column(
        dataframe,
        PAYMENT_METHOD_COLUMN_OPTIONS,
        required=False,
    )

    if not payment_column:
        return pd.DataFrame()

    return (
        dataframe.groupby(
            payment_column,
            dropna=False,
        )
        .agg(
            Total_Spending=(
                amount_column,
                "sum",
            ),
            Transaction_Count=(
                amount_column,
                "count",
            ),
            Average_Transaction=(
                amount_column,
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Total_Spending",
            ascending=False,
        )
        .round(2)
    )


# ---------------------------------------------------------
# Budget analysis
# ---------------------------------------------------------

def analyze_budget_status(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Compare budgeted and unbudgeted spending."""
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    budgeted_column = find_column(
        dataframe,
        BUDGETED_COLUMN_OPTIONS,
        required=False,
    )

    if not budgeted_column:
        return {
            "available": False,
            "message": (
                "The dataset does not contain "
                "a Budgeted column."
            ),
            "table": pd.DataFrame(),
        }

    table = (
        dataframe.groupby(
            budgeted_column
        )
        .agg(
            Total_Spending=(
                amount_column,
                "sum",
            ),
            Transaction_Count=(
                amount_column,
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "Total_Spending",
            ascending=False,
        )
        .round(2)
    )

    unbudgeted_spending = float(
        dataframe.loc[
            dataframe[budgeted_column] == "No",
            amount_column,
        ].sum()
    )

    return {
        "available": True,
        "table": table,
        "unbudgeted_spending": unbudgeted_spending,
    }

# ---------------------------------------------------------
# Income analysis
# ---------------------------------------------------------

def analyze_income(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compare monthly spending with Monthly_Income_USD.

    Monthly income appears repeatedly on transaction rows, so the
    analysis uses one income value per month rather than summing it.
    """
    amount_column = find_column(
        dataframe,
        AMOUNT_COLUMN_OPTIONS,
    )

    income_column = find_column(
        dataframe,
        INCOME_COLUMN_OPTIONS,
        required=False,
    )

    if not income_column:
        return {
            "available": False,
            "message": (
                "The dataset does not contain "
                "a Monthly_Income_USD column."
            ),
            "monthly": pd.DataFrame(),
        }

    date_column = find_column(
        dataframe,
        DATE_COLUMN_OPTIONS,
        required=False,
    )

    working = dataframe.dropna(
        subset=[income_column]
    ).copy()

    if working.empty:
        return {
            "available": False,
            "message": (
                "Monthly income values are missing."
            ),
            "monthly": pd.DataFrame(),
        }

    if date_column:
        working = working.dropna(
            subset=[date_column]
        ).copy()

        working["Analysis_Month"] = (
            working[date_column]
            .dt.to_period("M")
            .astype(str)
        )
    else:
        month_column = find_column(
            working,
            MONTH_COLUMN_OPTIONS,
            required=False,
        )

        if not month_column:
            return {
                "available": False,
                "message": (
                    "A date or month column is needed "
                    "for income comparison."
                ),
                "monthly": pd.DataFrame(),
            }

        working["Analysis_Month"] = (
            working[month_column].astype(str)
        )

    monthly = (
        working.groupby(
            "Analysis_Month"
        )
        .agg(
            Monthly_Spending=(
                amount_column,
                "sum",
            ),
            Monthly_Income=(
                income_column,
                "first",
            ),
        )
        .reset_index()
        .sort_values(
            "Analysis_Month"
        )
    )

    monthly["Remaining_Income"] = (
        monthly["Monthly_Income"]
        - monthly["Monthly_Spending"]
    )

    monthly["Spending_to_Income_Percentage"] = (
        monthly["Monthly_Spending"]
        / monthly["Monthly_Income"]
        * 100
    )

    monthly = monthly.round(2)

    return {
        "available": True,
        "monthly": monthly,
        "average_monthly_income": float(
            monthly["Monthly_Income"].mean()
        ),
        "average_monthly_spending": float(
            monthly["Monthly_Spending"].mean()
        ),
        "average_spending_ratio": float(
            monthly[
                "Spending_to_Income_Percentage"
            ].mean()
        ),
    }

# ---------------------------------------------------------
# Savings opportunities
# ---------------------------------------------------------

def calculate_savings_opportunities(
    dataframe: pd.DataFrame,
    reduction_percentage: float = 20,
) -> dict[str, Any]:
    """Estimate savings from reducing discretionary spending."""
    needs_wants = analyze_needs_and_wants(
        dataframe
    )
    impulse = analyze_impulse_spending(
        dataframe
    )

    want_spending = (
        float(
            needs_wants.get(
                "want_spending",
                0.0,
            )
        )
        if needs_wants.get("available")
        else 0.0
    )

    impulse_spending = (
        float(
            impulse.get(
                "total_impulse_spending",
                0.0,
            )
        )
        if impulse.get("available")
        else 0.0
    )

    # Potential impulse spending is normally a subset of wants,
    # so do not add both figures together and double-count them.
    discretionary_base = max(
        want_spending,
        impulse_spending,
    )

    estimated_savings = (
        discretionary_base
        * reduction_percentage
        / 100
    )

    return {
        "reduction_percentage": reduction_percentage,
        "want_spending": want_spending,
        "impulse_spending": impulse_spending,
        "discretionary_base": discretionary_base,
        "estimated_savings_opportunity": (
            estimated_savings
        ),
    }
# ---------------------------------------------------------
# Generate insights
# ---------------------------------------------------------

def generate_spending_insights(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Generate understandable spending observations."""
    insights: list[str] = []

    summary = calculate_basic_summary(
        dataframe
    )
    categories = analyze_categories(
        dataframe
    )
    needs_wants = analyze_needs_and_wants(
        dataframe
    )
    impulse = analyze_impulse_spending(
        dataframe
    )
    monthly = analyze_monthly_spending(
        dataframe
    )
    budget = analyze_budget_status(
        dataframe
    )
    income = analyze_income(
        dataframe
    )

    insights.append(
        f"The dataset contains "
        f"{summary['transaction_count']} spending transactions "
        f"with total spending of "
        f"{money(summary['total_spending'])}."
    )

    insights.append(
        "The average transaction amount is "
        f"{money(summary['average_transaction'])}."
    )

    if not categories.empty:
        category_column = find_column(
            dataframe,
            CATEGORY_COLUMN_OPTIONS,
        )
        top_category = categories.iloc[0]

        insights.append(
            f"The highest-spending category is "
            f"{top_category[category_column]}, representing "
            f"{top_category['Percentage_of_Total']:.2f}% "
            "of total spending."
        )

    if needs_wants.get("available"):
        percentage = float(
            needs_wants["want_percentage"]
        )

        insights.append(
            "Want spending totals "
            f"{money(needs_wants['want_spending'])}, "
            f"which is {percentage:.2f}% "
            "of total spending."
        )

        if percentage >= 40:
            insights.append(
                "Want spending forms a large share of spending. "
                "Reviewing discretionary purchases could create "
                "meaningful savings."
            )
        elif percentage >= 25:
            insights.append(
                "Want spending is moderate. A monthly limit for "
                "discretionary purchases may improve control."
            )
        else:
            insights.append(
                "Want spending is a relatively small share "
                "of total spending."
            )

    if impulse.get("available"):
        percentage = float(
            impulse[
                "impulse_spending_percentage"
            ]
        )

        insights.append(
            "Potential impulse spending totals "
            f"{money(impulse['total_impulse_spending'])}, "
            f"or {percentage:.2f}% of total spending."
        )

        if percentage >= 20:
            insights.append(
                "Potential impulse spending is high. A waiting "
                "period before nonessential purchases may help."
            )
        elif percentage >= 10:
            insights.append(
                "Potential impulse spending is noticeable. "
                "Tracking emotional and situational triggers "
                "may help reduce it."
            )
        else:
            insights.append(
                "Potential impulse spending appears limited."
            )

    if budget.get("available"):
        insights.append(
            "Unbudgeted spending totals "
            f"{money(budget['unbudgeted_spending'])}."
        )

    if income.get("available"):
        insights.append(
            "Average monthly spending equals "
            f"{income['average_spending_ratio']:.2f}% "
            "of monthly income."
        )

    if not monthly.empty:
        highest_month = monthly.loc[
            monthly["Total_Spending"].idxmax()
        ]
        month_label = highest_month.iloc[0]

        insights.append(
            f"The highest-spending month is {month_label}, "
            f"with spending of "
            f"{money(highest_month['Total_Spending'])}."
        )

    return insights

# ---------------------------------------------------------
# Run complete analysis
# ---------------------------------------------------------

def analyze_spending(
    file_path: str | Path = DEFAULT_DATA_FILE,
) -> dict[str, Any]:
    """Run all spending analyses."""
    dataframe = load_spending_data(
        file_path
    )

    return {
        "dataframe": dataframe,
        "summary": calculate_basic_summary(
            dataframe
        ),
        "categories": analyze_categories(
            dataframe
        ),
        "needs_wants": analyze_needs_and_wants(
            dataframe
        ),
        "impulse": analyze_impulse_spending(
            dataframe
        ),
        "monthly": analyze_monthly_spending(
            dataframe
        ),
        "payment_methods": (
            analyze_payment_methods(
                dataframe
            )
        ),
        "budget": analyze_budget_status(
            dataframe
        ),
        "income": analyze_income(
            dataframe
        ),
        "savings": (
            calculate_savings_opportunities(
                dataframe,
                reduction_percentage=20,
            )
        ),
        "insights": generate_spending_insights(
            dataframe
        ),
    }

# ---------------------------------------------------------
# Display analysis in terminal
# ---------------------------------------------------------

def display_analysis(
    results: dict[str, Any],
) -> None:
    """Print the spending analysis."""
    summary = results["summary"]

    print("\n" + "=" * 76)
    print("AI PERSONAL FINANCE SPENDING ANALYSIS")
    print("=" * 76)

    print("\nGENERAL SUMMARY")
    print("-" * 76)
    print(
        f"Number of transactions: "
        f"{summary['transaction_count']}"
    )
    print(
        f"Total spending: "
        f"{money(summary['total_spending'])}"
    )
    print(
        f"Average transaction: "
        f"{money(summary['average_transaction'])}"
    )
    print(
        f"Median transaction: "
        f"{money(summary['median_transaction'])}"
    )
    print(
        f"Highest transaction: "
        f"{money(summary['highest_transaction'])}"
    )
    print(
        f"Lowest transaction: "
        f"{money(summary['lowest_transaction'])}"
    )

    print("\nSPENDING BY CATEGORY")
    print("-" * 76)
    print(
        results["categories"].to_string(
            index=False
        )
    )

    print("\nNEEDS VERSUS WANTS")
    print("-" * 76)
    needs_wants = results["needs_wants"]

    if needs_wants.get("available"):
        print(
            needs_wants["table"].to_string(
                index=False
            )
        )
        print(
            "\nNeed spending: "
            f"{money(needs_wants['need_spending'])}"
        )
        print(
            "Want spending: "
            f"{money(needs_wants['want_spending'])}"
        )
    else:
        print(needs_wants["message"])

    print("\nPOTENTIAL IMPULSE SPENDING")
    print("-" * 76)
    impulse = results["impulse"]

    if impulse.get("available"):
        print(
            "Potential impulse transactions: "
            f"{impulse['impulse_count']}"
        )
        print(
            "Potential impulse spending: "
            f"{money(impulse['total_impulse_spending'])}"
        )
        print(
            "Percentage of total spending: "
            f"{impulse['impulse_spending_percentage']:.2f}%"
        )

        if not impulse["by_category"].empty:
            print(
                "\nImpulse spending by category:"
            )
            print(
                impulse["by_category"].to_string(
                    index=False
                )
            )
    else:
        print(impulse["message"])

    print("\nMONTHLY SPENDING")
    print("-" * 76)
    monthly = results["monthly"]

    if monthly.empty:
        print(
            "No date or month information was available."
        )
    else:
        print(
            monthly.to_string(
                index=False
            )
        )

    print("\nPAYMENT METHODS")
    print("-" * 76)
    payment_methods = results[
        "payment_methods"
    ]

    if payment_methods.empty:
        print(
            "No payment-method information was available."
        )
    else:
        print(
            payment_methods.to_string(
                index=False
            )
        )

    print("\nBUDGETED VERSUS UNBUDGETED")
    print("-" * 76)
    budget = results["budget"]

    if budget.get("available"):
        print(
            budget["table"].to_string(
                index=False
            )
        )
        print(
            "\nUnbudgeted spending: "
            f"{money(budget['unbudgeted_spending'])}"
        )
    else:
        print(budget["message"])

    print("\nINCOME COMPARISON")
    print("-" * 76)
    income = results["income"]

    if income.get("available"):
        print(
            income["monthly"].to_string(
                index=False
            )
        )
        print(
            "\nAverage spending-to-income ratio: "
            f"{income['average_spending_ratio']:.2f}%"
        )
    else:
        print(income["message"])

    print("\nSAVINGS OPPORTUNITY")
    print("-" * 76)
    savings = results["savings"]

    print(
        "Estimated savings from reducing "
        f"discretionary spending by "
        f"{savings['reduction_percentage']:.0f}%: "
        f"{money(savings['estimated_savings_opportunity'])}"
    )

    print("\nKEY INSIGHTS")
    print("-" * 76)

    for number, insight in enumerate(
        results["insights"],
        start=1,
    ):
        print(
            f"{number}. {insight}"
        )

    print("\n" + "=" * 76)


# ---------------------------------------------------------
# Run directly
# ---------------------------------------------------------

if __name__ == "__main__":
    try:
        analysis_results = analyze_spending()
        display_analysis(
            analysis_results
        )

    except Exception as error:
        print(
            "\nThe spending analysis could not run."
        )
        print(
            f"\nError: {error}"
        )
        print(
            "\nCheck that:"
            "\n1. Your virtual environment is active."
            "\n2. pandas and openpyxl are installed."
            "\n3. The Excel file is inside the data folder."
            "\n4. The Excel filename is correct."
            "\n5. The dataset contains Amount_USD."
        )