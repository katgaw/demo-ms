# Margin Loan Client Policy

This document defines logical rules that margin loan clients must satisfy. All thresholds apply per borrower at assessment time.

## Collateral and leverage

1. **Loan-to-value (LTV) cap**: The client's `ltv_ratio` must be less than or equal to **1.0**. Values strictly greater than 1.0 violate policy.

2. **Maintenance margin**: The client's `margin_percentage` must be greater than or equal to **0.35** (35% equity requirement).

3. **Collateral coverage**: The client's `collateral_value` must be strictly greater than zero whenever `loan_amount` is greater than zero.

## Credit and limits

4. **Credit limit**: The `loan_amount` must not exceed the client's `credit_limit`.

5. **Payment history**: The `payment_history_score` must be greater than or equal to **60**.

## Legal and liquidity

6. **Legal risk**: The `legal_risk_flag` must equal **0**. If `legal_risk_flag` is **1**, the client fails legal eligibility.

7. **Liquidity**: The client's `liquidity` must be strictly greater than **0**.

## Income and stress (advisory checks)

8. **Income presence**: The client's `income` must be strictly greater than **0** for new margin extensions.

9. **Market stress**: If `market_drop_sensitivity` is greater than **0.35**, the relationship must be flagged for enhanced monitoring (still a logical rule: sensitivity above this threshold requires documented exception or restructuring).

Use only the variable names above (`borrower_id`, `income`, `net_worth`, `liquidity`, `credit_limit`, `loan_amount`, `collateral_value`, `margin_percentage`, `margin_value`, `ltv_ratio`, `asset_volatility`, `market_drop_sensitivity`, `interest_rate`, `payment_history_score`, `legal_risk_flag`) when mapping rules to client data.
