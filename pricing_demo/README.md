# RM6 & AirSeT quote simulator

This Streamlit demo implements the joint pricing model in slides 6–10 of **BOM Cost and Pricing - Licheng(1).pptx**.

## Run locally

From this folder, run the following command with `uv` installed. The first launch installs the dependencies.

```powershell
uv run --with-requirements requirements.txt streamlit run app.py
```

With an existing Python environment, use these commands instead.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Live demo

1. Open **Presentation presets** and select **Slide baseline · 3 months**. Buyer A takes RM6 at €12,249.99 and Buyer B takes AirSeT at €12,369.98.
2. Select **Longer delivery · 6 months**. Both buyers take RM6. At the same RM6 quote, the seller needs a premium of €336.21 to sell AirSeT while satisfying both scenarios. Buyer B accepts at most €119.99.
3. Raise Buyer B’s **W (€)** to €400. The optimum offers both products again, with an AirSeT quote of €12,649.98.
4. Choose **Test a fixed quote** to hold the quotes while editing either buyer’s valuations. Set RM6 to €12,250 for a buyer with `B = €12,250` to demonstrate the zero-surplus rule when AirSeT is unavailable or also yields no positive surplus.

## Pricing rules

- `B` is the buyer’s willingness to pay for RM6. `W` is the extra willingness to pay for AirSeT. Eligible offers are compared using `B − Pᵣ` and `B + W − Pₐ`. RM6 wins a positive tie, and the buyer walks away at zero surplus.
- The optimizer compares the two single-product menus and a joint menu, with a no-sale result if no feasible quote attracts a buyer. It optimizes worst-case priced-BOM contribution, which is P90 here because both P90 costs exceed their P50 costs. Both states use the same quote and buyer choices.
- Prices are searched in whole cents. Under both cost states, offered prices must be positive and meet the minimum priced-BOM rate. The maximum rate gap applies when both products are offered.
- The fixed-RM6 premium comparison requires a strict contribution improvement for the switch in both scenarios. The optimizer also considers other RM6 prices and single-product offers.
- Costs are from the `product_costs.csv` basis for slide 8, kept to four decimals for calculations. Contributions deduct priced BOM costs.
