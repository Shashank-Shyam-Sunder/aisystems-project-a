# Heatmap Analysis

## Key Observations

- Several categories (payments, returns, rewards, shipping, support, sustainability, troubleshooting) show reduced correctness (~60%) at hard difficulty.
- Performance generally declines as difficulty increases, indicating that the system struggles with more complex queries.
- Some categories (orders, pricing, promotions, warranty) perform consistently well across all difficulty levels, suggesting strong coverage and straightforward retrieval.
- Certain categories (support, business, products, sustainability, troubleshooting) also show lower correctness even at easy difficulty, indicating category-specific weaknesses beyond just query complexity.

## Main Failure Pattern

The system performs well on easy and medium queries but shows consistent degradation on hard queries. This suggests that retrieval is largely effective, but answer generation struggles with precision and completeness under more complex conditions.

Additionally, some categories exhibit weaknesses even at easy difficulty, indicating gaps in domain understanding or response formulation.

## Week 2 Priorities

- Improve performance on hard difficulty queries across multiple categories
- Focus specifically on weak categories:
  - support
  - troubleshooting
  - sustainability
  - payments
  - returns
- Reduce verbosity and improve answer precision to better match expected outputs