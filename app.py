import os, json
import pandas as pd
from flask import Flask, jsonify, send_from_directory, render_template

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, "outputs", "tables")
STITCH = os.path.join(BASE, "stitch_fashiondemand_ai_analytics_dashboard")

app = Flask(__name__, template_folder=os.path.join(BASE, "templates"), static_folder=os.path.join(BASE, "static"))


def read_csv(name):
    path = os.path.join(TABLES, f"{name}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def read_json(name):
    path = os.path.join(TABLES, f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------- PAGES ----------

@app.route("/")
def overview():
    return render_template("overview.html")

@app.route("/inventory")
def inventory():
    return render_template("inventory.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/models")
def models():
    return render_template("models.html")


# ---------- API ENDPOINTS ----------

@app.route("/api/summary")
def api_summary():
    inv = read_json("inventory_summary")
    metrics = read_json("model_metrics")
    country = read_csv("country_demand")
    category = read_csv("category_demand")
    monthly = read_csv("monthly_demand")

    total_revenue = sum(r.get("total_revenue", 0) for r in country) if country else 0
    total_units = sum(r.get("transaction_count", 0) for r in country) if country else 0
    rf = metrics.get("rf", {})

    return jsonify({
        "total_revenue": round(total_revenue, 0),
        "total_units": sum(r.get("total_units", 0) for r in category) if category else 0,
        "active_stores": 35,
        "countries": 7,
        "risk_segments": inv.get("stockout_risk", 0),
        "rf_r2": rf.get("r2", 0),
        "rf_rmse": rf.get("rmse", 0),
        "rf_mae": rf.get("mae", 0),
        "rf_smape": rf.get("smape", 0),
    })


@app.route("/api/country_demand")
def api_country_demand():
    return jsonify(read_csv("country_demand"))


@app.route("/api/store_demand")
def api_store_demand():
    return jsonify(read_csv("store_demand"))


@app.route("/api/monthly_demand")
def api_monthly_demand():
    return jsonify(read_csv("monthly_demand"))


@app.route("/api/category_demand")
def api_category_demand():
    return jsonify(read_csv("category_demand"))


@app.route("/api/subcategory_demand")
def api_subcategory_demand():
    return jsonify(read_csv("subcategory_demand"))


@app.route("/api/weekday_demand")
def api_weekday_demand():
    return jsonify(read_csv("weekday_demand"))


@app.route("/api/hourly_demand")
def api_hourly_demand():
    return jsonify(read_csv("hourly_demand"))


@app.route("/api/discount_analysis")
def api_discount_analysis():
    return jsonify(read_csv("discount_analysis"))


@app.route("/api/country_category")
def api_country_category():
    return jsonify(read_csv("country_category"))


@app.route("/api/top_products")
def api_top_products():
    return jsonify(read_csv("top_products"))


@app.route("/api/top_stores")
def api_top_stores():
    return jsonify(read_csv("top_stores"))


@app.route("/api/seasonal")
def api_seasonal():
    return jsonify(read_csv("seasonal"))


@app.route("/api/discount_impact")
def api_discount_impact():
    return jsonify(read_csv("discount_impact"))


@app.route("/api/payment_methods")
def api_payment_methods():
    return jsonify(read_csv("payment_methods"))


@app.route("/api/price_distribution")
def api_price_distribution():
    return jsonify(read_csv("price_distribution"))


@app.route("/api/revenue_per_customer")
def api_revenue_per_customer():
    return jsonify(read_csv("revenue_per_customer"))


@app.route("/api/yearly_growth")
def api_yearly_growth():
    return jsonify(read_csv("yearly_growth"))


@app.route("/api/transaction_type")
def api_transaction_type():
    return jsonify(read_csv("transaction_type"))


@app.route("/api/model_metrics")
def api_model_metrics():
    return jsonify(read_json("model_metrics"))


@app.route("/api/feature_importances")
def api_feature_importances():
    return jsonify(read_json("feature_importances"))


@app.route("/api/segment_metrics")
def api_segment_metrics():
    return jsonify(read_json("segment_metrics"))


@app.route("/api/inventory_summary")
def api_inventory_summary():
    return jsonify(read_json("inventory_summary"))


@app.route("/api/inventory_recommendations")
def api_inventory_recommendations():
    return jsonify(read_csv("inventory_recommendations"))


@app.route("/api/demand_statistics")
def api_demand_statistics():
    return jsonify(read_csv("demand_statistics"))


# ---------- CHARTS (static PNG files) ----------

@app.route("/charts/<path:filename>")
def charts(filename):
    return send_from_directory(os.path.join(BASE, "outputs", "charts"), filename)


if __name__ == "__main__":
    print("=" * 50)
    print("FashionDemand AI Dashboard")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000)
