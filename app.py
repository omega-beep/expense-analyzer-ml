from flask import Flask, render_template, request, send_file
import pandas as pd
import io
from predictor import predict_top_categories

app = Flask(__name__)

# store last CSV results for download
last_export_df = None

@app.route("/", methods=["GET", "POST"])
def index():
    global last_export_df

    predictions = None
    bulk_results = None
    category_summary = None
    error = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        # -------- SINGLE EXPENSE --------
        if form_type == "single":
            description = request.form.get("description", "").strip()

            if not description:
                error = "Please enter an expense description."
            else:
                predictions = predict_top_categories(description)

        # -------- CSV UPLOAD --------
        elif form_type == "csv":
            file = request.files.get("csv_file")

            try:
                df = pd.read_csv(file)

                if df.empty:
                    raise ValueError("CSV file is empty.")

                if "description" not in df.columns:
                    raise ValueError("CSV must contain a 'description' column.")

                bulk_results = []
                category_summary = {}
                export_rows = []

                for desc in df["description"]:
                    preds = predict_top_categories(desc)

                    # top-1 for summary
                    top1_cat, top1_conf = preds[0]
                    top2_cat, top2_conf = preds[1]

                    category_summary[top1_cat] = category_summary.get(top1_cat, 0) + 1

                    bulk_results.append({
                        "description": desc,
                        "predictions": preds
                    })

                    export_rows.append({
                        "description": desc,
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
        predictions=predictions,
        bulk_results=bulk_results,
        category_summary=category_summary,
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
        download_name="expense_predictions.csv"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8004, debug=True)
