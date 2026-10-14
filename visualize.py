import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

OUTPUTS = "outputs"
TABLES = os.path.join(OUTPUTS, "tables")
CHARTS = os.path.join(OUTPUTS, "charts")
os.makedirs(CHARTS, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DAYS = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}


def chart_country_demand():
    df = pd.read_csv(os.path.join(TABLES, "country_demand.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    axes[0].barh(df["country"], df["total_revenue"], color=colors[:len(df)])
    axes[0].set_xlabel("Total Revenue")
    axes[0].set_title("Revenue by Country")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    axes[1].barh(df["country"], df["transaction_count"], color=colors[:len(df)])
    axes[1].set_xlabel("Transaction Count")
    axes[1].set_title("Transactions by Country")
    plt.suptitle("Global Fashion Retail - Country Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "01_country_demand.png"))
    plt.close()
    print("  01_country_demand.png")


def chart_monthly_trend():
    df = pd.read_csv(os.path.join(TABLES, "monthly_demand.csv"))
    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df["period"], df["total_units"], "o-", color="#e74c3c", linewidth=2, markersize=4)
    axes[0].fill_between(df["period"], df["total_units"], alpha=0.15, color="#e74c3c")
    axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Monthly Demand Trend (Units)")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(df["period"], df["total_revenue"], color="#3498db", alpha=0.8)
    axes[1].set_ylabel("Revenue")
    axes[1].set_title("Monthly Revenue Trend")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    axes[1].grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.suptitle("Monthly Demand & Revenue Trend (2023-2025)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "02_monthly_trend.png"))
    plt.close()
    print("  02_monthly_trend.png")


def chart_category_demand():
    df = pd.read_csv(os.path.join(TABLES, "category_demand.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    axes[0].pie(df["total_units"], labels=df["category"], autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Units Sold by Category")
    axes[1].bar(df["category"], df["total_revenue"], color=colors)
    axes[1].set_ylabel("Revenue")
    axes[1].set_title("Revenue by Category")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    plt.suptitle("Category-Level Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "03_category_demand.png"))
    plt.close()
    print("  03_category_demand.png")


def chart_subcategory_demand():
    df = pd.read_csv(os.path.join(TABLES, "subcategory_demand.csv"))
    df = df.sort_values("total_revenue", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if c == "Masculine" else "#3498db" if c == "Feminine" else "#2ecc71"
              for c in df["category"]]
    ax.barh(df["sub_category"], df["total_revenue"], color=colors, alpha=0.85)
    ax.set_xlabel("Total Revenue")
    ax.set_title("Top 15 Sub-Categories by Revenue")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Masculine"),
                       Patch(facecolor="#3498db", label="Feminine"),
                       Patch(facecolor="#2ecc71", label="Children")]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "04_subcategory_demand.png"))
    plt.close()
    print("  04_subcategory_demand.png")


def chart_weekday_hourly():
    df_d = pd.read_csv(os.path.join(TABLES, "weekday_demand.csv"))
    df_h = pd.read_csv(os.path.join(TABLES, "hourly_demand.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_d["day_name"] = df_d["day_of_week"].map(DAYS)
    axes[0].bar(df_d["day_name"], df_d["total_units"], color="#e74c3c", alpha=0.8)
    axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Demand by Day of Week")
    axes[0].grid(True, alpha=0.3, axis="y")
    df_h["hour"] = df_h["hour_of_day"].astype(int)
    axes[1].bar(df_h["hour"], df_h["total_units"], color="#3498db", alpha=0.8)
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Units Sold")
    axes[1].set_title("Demand by Hour of Day")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.suptitle("Temporal Demand Patterns", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "05_weekday_hourly.png"))
    plt.close()
    print("  05_weekday_hourly.png")


def chart_discount_analysis():
    df = pd.read_csv(os.path.join(TABLES, "discount_analysis.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(df["has_discount"], df["total_units"], color=["#2ecc71", "#e74c3c"])
    axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Units: Full Price vs Discounted")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(df["has_discount"], df["avg_line_total"], color=["#2ecc71", "#e74c3c"])
    axes[1].set_ylabel("Avg Line Total ($)")
    axes[1].set_title("Avg Transaction Value")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.suptitle("Discount Impact Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "06_discount_analysis.png"))
    plt.close()
    print("  06_discount_analysis.png")


def chart_discount_impact():
    df = pd.read_csv(os.path.join(TABLES, "discount_impact.csv"))
    cats = df["category"].unique()
    x = range(len(cats))
    width = 0.35
    disc = df[df["has_discount"] == "Discounted"].set_index("category").reindex(cats)
    full = df[df["has_discount"] == "Full Price"].set_index("category").reindex(cats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([i - width/2 for i in x], disc["total_units"], width, label="Discounted", color="#e74c3c")
    axes[0].bar([i + width/2 for i in x], full["total_units"], width, label="Full Price", color="#3498db")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats)
    axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Units: Discounted vs Full Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar([i - width/2 for i in x], disc["avg_line_total"], width, label="Discounted", color="#e74c3c")
    axes[1].bar([i + width/2 for i in x], full["avg_line_total"], width, label="Full Price", color="#3498db")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].set_ylabel("Avg Line Total ($)")
    axes[1].set_title("Avg Value: Discounted vs Full Price")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.suptitle("Discount Impact by Category", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "07_discount_impact.png"))
    plt.close()
    print("  07_discount_impact.png")


def chart_top_stores():
    df = pd.read_csv(os.path.join(TABLES, "top_stores.csv"))
    df = df.head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = df["city"] + " (" + df["country"] + ")"
    colors = {"中国": "#e74c3c", "United States": "#3498db", "Deutschland": "#f39c12",
              "France": "#9b59b6", "Portugal": "#2ecc71", "España": "#e67e22",
              "United Kingdom": "#1abc9c"}
    bar_colors = [colors.get(c, "#95a5a6") for c in df["country"]]
    ax.barh(labels[::-1], df["total_revenue"][::-1], color=bar_colors[::-1])
    ax.set_xlabel("Total Revenue")
    ax.set_title("Top 15 Stores by Revenue")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "08_top_stores.png"))
    plt.close()
    print("  08_top_stores.png")


def chart_seasonal():
    df = pd.read_csv(os.path.join(TABLES, "seasonal.csv"))
    df["month_name"] = df["month"].map(MONTHS)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = "#e74c3c"
    color2 = "#3498db"
    ax1.bar(df["month_name"], df["total_units"], color=color1, alpha=0.7, label="Units")
    ax1.set_ylabel("Units Sold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(df["month_name"], df["total_revenue"], "o-", color=color2, linewidth=2, label="Revenue")
    ax2.set_ylabel("Revenue", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    fig.suptitle("Seasonal Demand Pattern by Month", fontsize=15, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "09_seasonal.png"))
    plt.close()
    print("  09_seasonal.png")


def chart_model_comparison():
    with open(os.path.join(TABLES, "model_metrics.json")) as f:
        metrics = json.load(f)
    rf = metrics["rf"]
    naive = {"rmse": 12.8654, "mae": 6.2397}
    models = ["Naive Baseline", "Random Forest"]
    rmses = [naive["rmse"], rf["rmse"]]
    maes = [naive["mae"], rf["mae"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#e74c3c", "#2ecc71"]
    axes[0].bar(models, rmses, color=colors)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE Comparison (lower = better)")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(rmses):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")
    axes[1].bar(models, maes, color=colors)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE Comparison (lower = better)")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")
    plt.suptitle("Forecasting Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "10_model_comparison.png"))
    plt.close()
    print("  10_model_comparison.png")

    info_text = f"""Random Forest Model Metrics:
  RMSE:  {rf['rmse']:.4f}
  MAE:   {rf['mae']:.4f}
  SMAPE: {rf['smape']:.2f}%
  R2:    {rf['r2']:.4f}
  N:     {rf['n']:,}

Naive Baseline (lag_1):
  RMSE: {naive['rmse']:.4f}
  MAE:  {naive['mae']:.4f}

Improvement over naive:
  RMSE reduction: {(1 - rf['rmse']/naive['rmse'])*100:.1f}%
  MAE reduction:  {(1 - rf['mae']/naive['mae'])*100:.1f}%
"""
    with open(os.path.join(OUTPUTS, "tables", "model_summary.txt"), "w") as f:
        f.write(info_text)


def chart_country_category():
    df = pd.read_csv(os.path.join(TABLES, "country_category.csv"))
    pivot = df.pivot_table(index="country", columns="category", values="total_units", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_ylabel("Units Sold")
    ax.set_title("Demand by Country and Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "11_country_category.png"))
    plt.close()
    print("  11_country_category.png")


def main():
    print("=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    chart_country_demand()
    chart_monthly_trend()
    chart_category_demand()
    chart_subcategory_demand()
    chart_weekday_hourly()
    chart_discount_analysis()
    chart_discount_impact()
    chart_top_stores()
    chart_seasonal()
    chart_model_comparison()
    chart_country_category()
    try:
        chart_inventory_risk()
        chart_inventory_by_store()
        chart_replenishment()
    except FileNotFoundError:
        print("  (Inventory charts skipped - run main_pipeline.py first)")
    print(f"\nAll charts saved to {CHARTS}/")


def chart_inventory_risk():
    import json
    with open(os.path.join(TABLES, "inventory_summary.json")) as f:
        s = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"Stockout Risk\n({s['stockout_risk']})", 
              f"Overstock Risk\n({s['overstock_risk']})",
              f"Healthy\n({s['healthy']})"]
    sizes = [s['stockout_risk'], s['overstock_risk'], s['healthy']]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                       autopct="%1.1f%%", startangle=90,
                                       explode=explode, textprops={"fontsize": 11})
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title(f"Inventory Risk Distribution ({s['total_segments']} segments)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "12_inventory_risk.png"))
    plt.close()
    print("  12_inventory_risk.png")


def chart_inventory_by_store():
    import pandas as pd
    df = pd.read_csv(os.path.join(TABLES, "inventory_recommendations.csv"))
    store_risk = df.groupby(["store_id", "city", "country"]).agg(
        stockout=("risk_status", lambda x: (x == "STOCKOUT_RISK").sum()),
        overstock=("risk_status", lambda x: (x == "OVERSTOCK_RISK").sum()),
        ok=("risk_status", lambda x: (x == "OK").sum()),
        total_order=("recommended_order_qty", "sum")
    ).reset_index().sort_values("stockout", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = range(len(store_risk))
    width = 0.35
    axes[0].barh([i - width/2 for i in x], store_risk["stockout"], width, 
                  label="Stockout Risk", color="#e74c3c", alpha=0.8)
    axes[0].barh([i + width/2 for i in x], store_risk["overstock"], width,
                  label="Overstock Risk", color="#f39c12", alpha=0.8)
    labels = store_risk["city"] + " (" + store_risk["country"] + ")"
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Number of Segments at Risk")
    axes[0].set_title("Risk Count by Store (Top 15)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(labels, store_risk["total_order"], color="#3498db", alpha=0.8)
    axes[1].set_xlabel("Recommended Order Quantity (units)")
    axes[1].set_title("Recommended Orders by Store (Top 15)")
    axes[1].grid(True, alpha=0.3, axis="x")
    plt.suptitle("Inventory Risk & Replenishment by Store", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "13_inventory_by_store.png"))
    plt.close()
    print("  13_inventory_by_store.png")


def chart_replenishment():
    import pandas as pd
    df = pd.read_csv(os.path.join(TABLES, "inventory_recommendations.csv"))
    urgency = df[df["recommended_order_qty"] > 0].groupby("risk_severity").agg(
        count=("recommended_order_qty", "count"),
        total_qty=("recommended_order_qty", "sum"),
        total_value=("estimated_stockout_date", "count")
    ).reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"CRITICAL": "#c0392b", "HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#f1c40f"}
    bar_colors = [colors.get(i, "#95a5a6") for i in urgency.index]
    
    axes[0].bar(urgency.index, urgency["count"], color=bar_colors)
    axes[0].set_ylabel("Number of Segments")
    axes[0].set_title("Urgency Distribution")
    for i, v in enumerate(urgency["count"]):
        axes[0].text(i, v + 1, str(int(v)), ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(urgency.index, urgency["total_qty"], color=bar_colors)
    axes[1].set_ylabel("Total Units to Order")
    axes[1].set_title("Order Volume by Urgency")
    for i, v in enumerate(urgency["total_qty"]):
        axes[1].text(i, v + 100, f"{int(v):,}", ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Replenishment Recommendations by Urgency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "14_replenishment.png"))
    plt.close()
    print("  14_replenishment.png")


if __name__ == "__main__":
    main()
