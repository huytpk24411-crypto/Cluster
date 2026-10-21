"""Đọc dữ liệu khách hàng từ MySQL thành pandas DataFrame.

Trước khi chạy, hãy import file
``datasets/salesdatabase/salesdatabase_customer.sql`` vào MySQL nếu cơ sở dữ
liệu ``salesdatabase`` chưa có sẵn.
"""

import os
import re
import sys
import base64
from io import BytesIO
from html import escape

import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# Hiển thị đúng tên khách hàng tiếng Việt khi chạy từ terminal Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def getConnect(server, port, database, username, password):
    """Tạo và trả về một kết nối MySQL; trả về None khi kết nối thất bại."""
    try:
        return mysql.connector.connect(
            host=server,
            port=port,
            database=database,
            user=username,
            password=password,
        )
    except Error as error:
        print("Không thể kết nối MySQL:", error)
        return None


def closeConnection(conn):
    """Đóng kết nối nếu kết nối đã được tạo."""
    if conn is not None and conn.is_connected():
        conn.close()


def queryDataset(conn, sql):
    """Thực thi truy vấn SELECT và trả kết quả dưới dạng DataFrame."""
    if conn is None or not conn.is_connected():
        raise ConnectionError("Chưa có kết nối MySQL hợp lệ.")

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=cursor.column_names)
    finally:
        cursor.close()


def export_df_to_excel(df, filename):
    """Xuất dữ liệu vào thư mục Results theo từng cụm kèm biểu đồ thống kê."""
    if "cluster" not in df.columns:
        raise ValueError('DataFrame phải có cột "cluster".')

    results_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
        "Results",
    )
    file_name = os.path.basename(os.fspath(filename))
    if not file_name:
        raise ValueError("filename phải chứa tên file Excel.")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, file_name)

    from openpyxl.chart import PieChart, Reference
    from openpyxl.chart.label import DataLabelList

    cluster_values = df["cluster"].drop_duplicates().tolist()
    missing_values = [value for value in cluster_values if pd.isna(value)]
    cluster_values = [value for value in cluster_values if not pd.isna(value)]
    try:
        cluster_values.sort()
    except TypeError:
        cluster_values.sort(key=lambda value: (type(value).__name__, str(value)))
    cluster_values.extend(missing_values)

    used_sheet_names = set()
    def make_sheet_name(cluster_value):
        label = "Thiếu dữ liệu" if pd.isna(cluster_value) else str(cluster_value)
        name = re.sub(r'[\\/*?:\[\]]', "_", f"Cluster {label}")[:31].strip("'")
        name = name or "Cluster"
        unique_name = name
        suffix = 2
        while unique_name.casefold() in used_sheet_names:
            ending = f"_{suffix}"
            unique_name = f"{name[:31 - len(ending)]}{ending}"
            suffix += 1
        used_sheet_names.add(unique_name.casefold())
        return unique_name

    summary_rows = []
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for cluster_value in cluster_values:
            if pd.isna(cluster_value):
                cluster_df = df[df["cluster"].isna()]
                summary_label = "Thiếu dữ liệu"
            else:
                cluster_df = df[df["cluster"] == cluster_value]
                summary_label = str(cluster_value)

            cluster_df.to_excel(
                writer,
                sheet_name=make_sheet_name(cluster_value),
                index=False,
            )
            summary_rows.append((summary_label, len(cluster_df)))

        if not summary_rows:
            summary_rows.append(("Không có dữ liệu", 0))

        summary_sheet_name = "Thống kê"
        pd.DataFrame(
            summary_rows,
            columns=["Cluster", "Số lượng"],
        ).to_excel(writer, sheet_name=summary_sheet_name, index=False)

        summary_sheet = writer.sheets[summary_sheet_name]
        chart = PieChart()
        chart.title = "Phân bố dữ liệu theo cụm"
        chart_data = Reference(
            summary_sheet,
            min_col=2,
            min_row=1,
            max_row=len(summary_rows) + 1,
        )
        chart_labels = Reference(
            summary_sheet,
            min_col=1,
            min_row=2,
            max_row=len(summary_rows) + 1,
        )
        chart.add_data(chart_data, titles_from_data=True)
        chart.set_categories(chart_labels)
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showPercent = True
        summary_sheet.add_chart(chart, "D2")

    return filename


def export_df_to_html(df, filename):
    """Xuất báo cáo HTML phân cụm khách hàng vào thư mục Results."""
    required_columns = {"cluster", "Age", "Annual Income", "Spending Score"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame thiếu cột: {', '.join(sorted(missing_columns))}.")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Results",
    )
    os.makedirs(results_dir, exist_ok=True)
    file_name = os.path.basename(os.fspath(filename))
    if not file_name:
        raise ValueError("filename phải chứa tên file HTML.")
    if not file_name.lower().endswith(".html"):
        file_name += ".html"
    output_path = os.path.join(results_dir, file_name)

    report_df = df.copy()
    report_df["Nhóm tuổi"] = pd.cut(
        report_df["Age"],
        bins=[-np.inf, 35, 45, np.inf],
        labels=["Thanh niên (≤35)", "Trung niên (36–45)", "Người cao tuổi (>45)"],
        include_lowest=True,
    )

    cluster_values = sorted(report_df["cluster"].dropna().unique().tolist())
    if report_df["cluster"].isna().any():
        cluster_values.append("Thiếu dữ liệu")

    def figure_as_base64(figure):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    cluster_labels = [str(value) for value in cluster_values]
    cluster_counts = [
        int((report_df["cluster"] == value).sum())
        if value != "Thiếu dữ liệu"
        else int(report_df["cluster"].isna().sum())
        for value in cluster_values
    ]
    pie_figure, pie_axis = plt.subplots(figsize=(7, 5))
    pie_axis.pie(
        cluster_counts,
        labels=[f"Cụm {label}" for label in cluster_labels],
        autopct="%1.1f%%",
        startangle=90,
    )
    pie_axis.set_title("Tỷ lệ khách hàng theo cụm")
    pie_axis.axis("equal")
    pie_chart = figure_as_base64(pie_figure)

    age_colors = {
        "Thanh niên (≤35)": "#2E86DE",
        "Trung niên (36–45)": "#F39C12",
        "Người cao tuổi (>45)": "#C0392B",
    }
    sections = []
    for cluster_value, cluster_label, cluster_count in zip(
        cluster_values, cluster_labels, cluster_counts
    ):
        if cluster_value == "Thiếu dữ liệu":
            cluster_df = report_df[report_df["cluster"].isna()].copy()
        else:
            cluster_df = report_df[report_df["cluster"] == cluster_value].copy()

        scatter_figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        for age_group, color in age_colors.items():
            group_df = cluster_df[cluster_df["Nhóm tuổi"] == age_group]
            axes[0].scatter(group_df["Age"], group_df["Annual Income"], label=age_group, color=color, alpha=0.8)
            axes[1].scatter(group_df["Age"], group_df["Spending Score"], label=age_group, color=color, alpha=0.8)
        axes[0].set_title(f"Cụm {cluster_label}: Độ tuổi và thu nhập")
        axes[0].set_xlabel("Độ tuổi")
        axes[0].set_ylabel("Annual Income")
        axes[1].set_title(f"Cụm {cluster_label}: Độ tuổi và Spending Score")
        axes[1].set_xlabel("Độ tuổi")
        axes[1].set_ylabel("Spending Score")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(title="Nhóm tuổi")
        scatter_figure.tight_layout()
        charts = figure_as_base64(scatter_figure)
        data_table = cluster_df.to_html(index=False, classes="data-table", border=0)
        sections.append(
            f'<section class="cluster-section"><h2>Cụm {escape(cluster_label)} — {cluster_count} khách hàng</h2>'
            f'<img src="data:image/png;base64,{charts}" alt="Biểu đồ cụm {escape(cluster_label)}">'
            f'<h3>Dữ liệu khách hàng trong cụm</h3><div class="table-container">{data_table}</div></section>'
        )

    page = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo phân cụm khách hàng</title><style>
body {{ font-family: Arial, sans-serif; margin: 32px auto; max-width: 1320px; color: #1f2937; background: #f8fafc; }}
h1,h2,h3 {{ color: #0f3d6e; }} .summary,.cluster-section {{ background:#fff; border-radius:10px; padding:24px; margin:22px 0; box-shadow:0 1px 5px #cbd5e1; }}
img {{ display:block; max-width:100%; height:auto; margin:14px auto; }} .table-container {{ max-height:440px; overflow:auto; border:1px solid #cbd5e1; }}
.data-table {{ border-collapse:collapse; width:100%; background:white; }} .data-table th {{ background:#0f3d6e; color:white; position:sticky; top:0; }}
.data-table th,.data-table td {{ padding:9px 12px; text-align:left; border-bottom:1px solid #e2e8f0; }} .data-table tr:nth-child(even) {{ background:#f8fafc; }}
</style></head><body><h1>Báo cáo phân cụm khách hàng</h1>
<div class="summary"><h2>Phân bố tỷ lệ khách hàng theo cụm</h2><img src="data:image/png;base64,{pie_chart}" alt="Tỷ lệ khách hàng theo cụm">
<p>Quy ước độ tuổi: <strong>Thanh niên ≤ 35</strong>; <strong>Trung niên từ 36 đến 45</strong>; <strong>Người cao tuổi &gt; 45</strong>.</p></div>
{''.join(sections)}</body></html>'''
    with open(output_path, "w", encoding="utf-8") as html_file:
        html_file.write(page)
    return output_path


def showHistogram(df, columns):
    """Vẽ histogram cho các cột số được chọn trong DataFrame."""
    plt.figure(figsize=(7, 8))
    for index, column in enumerate(columns, start=1):
        plt.subplot(len(columns), 1, index)
        sns.histplot(df[column], bins=32, kde=True)
        plt.title(f"Histogram of {column}")
    plt.tight_layout()
    plt.show()


def elbowMethod(df, columnsForElbow):
    """Vẽ elbow và tự động trả về số cụm tối ưu.

    Elbow được xác định bằng khoảng cách vuông góc lớn nhất từ mỗi điểm
    inertia đến đường nối điểm đầu-cuối. Nếu nhiều điểm có độ cong gần bằng
    nhau (chênh không quá 5%), chọn K lớn hơn.
    """
    x = df.loc[:, columnsForElbow].values
    inertia = []
    cluster_counts = np.arange(1, 11)

    for count in cluster_counts:
        model = KMeans(
            n_clusters=count,
            init="k-means++",
            max_iter=500,
            random_state=42,
            n_init=10,
        )
        model.fit(x)
        inertia.append(model.inertia_)

    # Chuẩn hóa hai trục rồi tìm điểm cách xa nhất đường nối đầu-cuối.
    # Điểm này là vị trí "gãy" của đường elbow.
    normalized_k = (cluster_counts - cluster_counts.min()) / (
        cluster_counts.max() - cluster_counts.min()
    )
    normalized_inertia = (np.array(inertia) - min(inertia)) / (
        max(inertia) - min(inertia)
    )
    line_start = np.array([normalized_k[0], normalized_inertia[0]])
    line_end = np.array([normalized_k[-1], normalized_inertia[-1]])
    line_vector = line_end - line_start
    points = np.column_stack((normalized_k, normalized_inertia))
    offsets = points - line_start
    distances = np.abs(
        line_vector[0] * offsets[:, 1] - line_vector[1] * offsets[:, 0]
    ) / np.linalg.norm(line_vector)

    max_distance = distances.max()
    near_elbows = cluster_counts[distances >= max_distance * 0.95]
    optimal_k = int(near_elbows.max())

    plt.figure(figsize=(10, 5))
    plt.plot(cluster_counts, inertia, "o-", alpha=0.7)
    plt.axvline(
        optimal_k,
        color="red",
        linestyle="--",
        label=f"Optimal K = {optimal_k}",
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("Cluster sum of squared distances")
    plt.title(f"Elbow method: {' × '.join(columnsForElbow)}")
    plt.xticks(np.arange(1, 11))
    plt.legend()
    plt.tight_layout()
    plt.show()
    return optimal_k


def runKMeans(x, cluster):
    """Huấn luyện K-Means và trả về nhãn cùng tọa độ tâm cụm."""
    model = KMeans(
        n_clusters=cluster,
        init="k-means++",
        max_iter=500,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(x)
    return labels, model.cluster_centers_, labels


def visualizeKMeans(x, y_kmeans, cluster, title, xlabel, ylabel, colors):
    """Trực quan hóa K-Means cho dữ liệu gồm đúng hai thuộc tính."""
    if x.shape[1] != 2:
        raise ValueError("visualizeKMeans chỉ hỗ trợ dữ liệu gồm hai cột.")
    if cluster > len(colors):
        raise ValueError("Số màu phải lớn hơn hoặc bằng số cụm.")

    plt.figure(figsize=(10, 8))
    for index in range(cluster):
        points = x[y_kmeans == index]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=70,
            c=colors[index],
            label=f"Cluster {index + 1}",
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Có thể đặt biến môi trường MYSQL_PASSWORD để dùng mật khẩu khác.
    conn = getConnect(
        server="localhost",
        port=3306,
        database="salesdatabase",
        username="root",
        password=os.getenv("MYSQL_PASSWORD", "@Obama123"),
    )

    try:
        customers = queryDataset(conn, "SELECT * FROM customer")
        print(customers)

        sql2 = """
            SELECT DISTINCT customer.CustomerID, customer.Age,
                   customer_spend_score.Annual_Income,
                   customer_spend_score.Spending_Score
            FROM customer
            INNER JOIN customer_spend_score
                ON customer.CustomerID = customer_spend_score.CustomerID
        """
        df2 = queryDataset(conn, sql2)
        df2.columns = ["CustomerId", "Age", "Annual Income", "Spending Score"]
        print(df2.head())
        print(df2.describe())
        showHistogram(df2, df2.columns[1:])

        colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]

        # Thí nghiệm 1: Age và Spending Score.
        columns = ["Age", "Spending Score"]
        cluster = elbowMethod(df2, columns)
        x = df2.loc[:, columns].values
        y_kmeans, centroids, labels = runKMeans(x, cluster)
        df2["Cluster Age-Spending"] = labels
        age_spending_export = df2.drop(
            columns=["Cluster Age-Spending", "Cluster Income-Spending"],
            errors="ignore",
        ).copy()
        age_spending_export["cluster"] = labels
        print(
            "Đã xuất dữ liệu Age-Spending:",
            export_df_to_excel(
                age_spending_export,
                "Customer_Clusters_Age_Spending.xlsx",
            ),
        )
        print(
            "Đã xuất báo cáo HTML Age-Spending:",
            export_df_to_html(
                age_spending_export,
                "Customer_Clusters_Age_Spending.html",
            ),
        )
        print("Tâm cụm Age × Spending Score:\n", centroids)
        visualizeKMeans(
            x,
            y_kmeans,
            cluster,
            "Clusters of Customers - Age × Spending Score",
            "Age",
            "Spending Score",
            colors,
        )

        # Thí nghiệm 2: Annual Income và Spending Score.
        columns = ["Annual Income", "Spending Score"]
        cluster = elbowMethod(df2, columns)
        x = df2.loc[:, columns].values
        y_kmeans, centroids, labels = runKMeans(x, cluster)
        df2["Cluster Income-Spending"] = labels
        income_spending_export = df2.drop(
            columns=["Cluster Age-Spending", "Cluster Income-Spending"],
            errors="ignore",
        ).copy()
        income_spending_export["cluster"] = labels
        print(
            "Đã xuất dữ liệu Income-Spending:",
            export_df_to_excel(
                income_spending_export,
                "Customer_Clusters_Income_Spending.xlsx",
            ),
        )
        print(
            "Đã xuất báo cáo HTML Income-Spending:",
            export_df_to_html(
                income_spending_export,
                "Customer_Clusters_Income_Spending.html",
            ),
        )
        print("Tâm cụm Annual Income × Spending Score:\n", centroids)
        visualizeKMeans(
            x,
            y_kmeans,
            cluster,
            "Clusters of Customers - Annual Income × Spending Score",
            "Annual Income",
            "Spending Score",
            colors,
        )
    finally:
        closeConnection(conn)
