from flask import Flask, render_template, request, send_file
import pandas as pd
import io
from predictor import predict_top_categories

app = Flask(__name__)

last_export_df = None

@app.route("/", methods=["GET", "POST"])
def index():
    global last_export_df

    single_result = None
    bulk_results = None
    category_summary = None
    amount_summary = None
    error = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        # -------- SINGLE EXPENSE --------
        if form_type == "single":
            description = request.form.get("description", "").strip()
            amount = request.form.get("amount", "").strip()

            if not description or not amount:
                error = "Please enter both description and amount."
            else:
                try:
                    amount = float(amount)
                    preds = predict_top_categories(description)
                    top_category, confidence = preds[0]

                    single_result = {
                        "description": description,
                        "amount": amount,
                        "category": top_category,
                        "confidence": confidence,
                        "predictions": preds
                    }
                except ValueError:
                    error = "Amount must be a valid number."

        # -------- CSV UPLOAD --------
        elif form_type == "csv":
            file = request.files.get("csv_file")

            try:
                df = pd.read_csv(file)

                required_cols = {"description", "amount"}
                if not required_cols.issubset(df.columns):
                    raise ValueError("CSV must contain 'description' and 'amount' columns.")

                bulk_results = []
                category_summary = {}
                amount_summary = {}
                export_rows = []

                for _, row in df.iterrows():
                    desc = str(row["description"])
                    amount = float(row["amount"])

                    preds = predict_top_categories(desc)
                    top1_cat, top1_conf = preds[0]
                    top2_cat, top2_conf = preds[1]

                    category_summary[top1_cat] = category_summary.get(top1_cat, 0) + 1
                    amount_summary[top1_cat] = amount_summary.get(top1_cat, 0) + amount

                    bulk_results.append({
                        "description": desc,
                        "amount": amount,
                        "predictions": preds
                    })

                    export_rows.append({
                        "description": desc,
                        "amount": amount,
                        "top_category": top1_cat,
                        "top_confidence": top1_conf,
                        "second_category": top2_cat,
                        "second_confidence": top2_conf
                    })

                last_export_df = pd.DataFrame(export_rows)

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        single_result=single_result,
        bulk_results=bulk_results,
        category_summary=category_summary,
        amount_summary=amount_summary,
        error=error
    )


@app.route("/download")
def download_csv():
    global last_export_df

    if last_export_df is None:
        return "No data available for export", 400

    output = io.StringIO()
    last_export_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="expense_analysis.csv"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8004, debug=True)
