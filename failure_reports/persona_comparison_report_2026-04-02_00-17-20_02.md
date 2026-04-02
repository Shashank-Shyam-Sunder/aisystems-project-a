# Persona Comparison Report

## Document  
07_promotional_events.md

## Summary Table

| Persona     | Hit Rate | MRR | Faithfulness | Correctness |
|------------|----------|-----|--------------|-------------|
| standard   | 1.00 | 0.90 | 5.00 | 3.40 |
| frustrated | 1.00 | 0.90 | 5.00 | 3.40 |
| mismatch   | 1.00 | 1.00 | 5.00 | 3.40 |

## Worst Performing Persona  
No clear worst persona

## Explanation  

All three personas perform almost identically across key metrics:
- Perfect hit rate (1.0)
- Perfect faithfulness (5.0)
- Identical correctness (3.4)

Mismatch shows slightly higher MRR, but this does not translate into better correctness.

A deeper look at evaluation results shows that the model consistently produces verbose answers that go beyond the expected response. While factually correct, these answers include extra details (e.g., premium policies, extended explanations), leading to partial correctness scores instead of full matches.

This pattern is consistent across all personas, indicating that the limitation is not query style but answer generation behavior.

## Conclusion  

Persona variation did not affect correctness because answer generation behavior dominated performance.
