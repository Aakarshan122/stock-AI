import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def _make_chart_actual_vs_pred(avp, symbol):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dates = avp['dates']
    actual = avp['actual']
    predicted = avp['predicted']
    ax.plot(dates, actual, color='#00e676', linewidth=2, label='Actual', marker='o', markersize=3)
    ax.plot(dates, predicted, color='#ff9800', linewidth=2, label='Predicted', linestyle='--', marker='x', markersize=3)
    ax.set_title(f'{symbol} — Actual vs Predicted Price (Test Set)', color='#333', fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('#ffffff')
    step = max(1, len(dates) // 6)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha='right')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ddd')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_chart_feature_importance(feat_importance, symbol):
    items = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels[::-1], values[::-1], color='#1976d2', edgecolor='#0d47a1')
    ax.set_title(f'{symbol} — Feature Importance (Random Forest)', color='#333', fontsize=11)
    ax.set_xlabel('Importance', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ddd')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_chart_confusion(clf_report, symbol):
    labels = ['DOWN (0)', 'UP (1)']
    tp = clf_report.get('1', {}).get('precision', 0)
    tn = clf_report.get('0', {}).get('precision', 0)
    fp = 1 - tn
    fn = 1 - tp
    matrix = np.array([[round(tn, 2), round(fp, 2)], [round(fn, 2), round(tp, 2)]])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred DOWN', 'Pred UP'], fontsize=8)
    ax.set_yticklabels(['Actual DOWN', 'Actual UP'], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=11,
                    color='white' if matrix[i, j] > 0.5 else 'black')
    ax.set_title(f'{symbol} — Precision Matrix', fontsize=10, color='#333')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(model_data):
    symbol = model_data['symbol']
    metrics = model_data['metrics']
    feat_importance = model_data['feat_importance']
    avp = model_data['actual_vs_pred']
    clf_report = metrics['clf_report']
    trained_at = model_data['trained_at']

    filename = f"{symbol.replace('.', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', fontSize=20, textColor=colors.HexColor('#1565c0'),
                                  alignment=TA_CENTER, spaceAfter=6, fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=11, textColor=colors.HexColor('#555555'),
                                alignment=TA_CENTER, spaceAfter=4)
    h2_style = ParagraphStyle('h2', fontSize=13, textColor=colors.HexColor('#1976d2'),
                               spaceAfter=6, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('body', fontSize=9, textColor=colors.HexColor('#333333'),
                                 spaceAfter=4, leading=14)

    story = []

    # Header
    story.append(Paragraph(f"📊 Stock AI Model Report", title_style))
    story.append(Paragraph(f"Symbol: <b>{symbol}</b> &nbsp;|&nbsp; Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style))
    story.append(Paragraph(f"Model Trained At: {trained_at}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1976d2'), spaceAfter=10))

    # Model Overview
    story.append(Paragraph("1. Model Overview", h2_style))
    overview_data = [
        ['Parameter', 'Value'],
        ['Stock Symbol', symbol],
        ['Model Type', 'Random Forest Classifier + Linear Regression'],
        ['Training Samples', str(metrics['train_size'])],
        ['Test Samples', str(metrics['test_size'])],
        ['Total Data Points', str(metrics['total_samples'])],
        ['Trained At', trained_at],
    ]
    t = Table(overview_data, colWidths=[7*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Performance Metrics
    story.append(Paragraph("2. Model Performance Metrics", h2_style))
    acc = metrics['accuracy']
    acc_color = '#2e7d32' if acc >= 60 else '#f57f17' if acc >= 50 else '#c62828'
    metrics_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Direction Accuracy', f"{acc}%",
         '✅ Good' if acc >= 60 else '⚠️ Average' if acc >= 50 else '❌ Low'],
        ['Mean Absolute Error (MAE)', str(metrics['mae']),
         'Lower is better — avg price prediction error'],
        ['R² Score', str(metrics['r2']),
         '✅ Strong fit' if metrics['r2'] >= 0.85 else '⚠️ Moderate' if metrics['r2'] >= 0.6 else '❌ Weak fit'],
        ['Precision (UP)', str(round(clf_report.get('1', {}).get('precision', 0), 3)), 'When model says UP, how often correct'],
        ['Recall (UP)', str(round(clf_report.get('1', {}).get('recall', 0), 3)), 'How many actual UPs were caught'],
        ['F1-Score (UP)', str(round(clf_report.get('1', {}).get('f1-score', 0), 3)), 'Balance of precision & recall'],
        ['Precision (DOWN)', str(round(clf_report.get('0', {}).get('precision', 0), 3)), 'When model says DOWN, how often correct'],
        ['Recall (DOWN)', str(round(clf_report.get('0', {}).get('recall', 0), 3)), 'How many actual DOWNs were caught'],
    ]
    t2 = Table(metrics_data, colWidths=[5.5*cm, 3*cm, 8.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # How model works in project
    story.append(Paragraph("3. How the Model Works in This Project", h2_style))
    how_text = """
    <b>Step 1 — Data Collection:</b> yfinance se last 2 years ka OHLCV data fetch hota hai.<br/>
    <b>Step 2 — Feature Engineering:</b> 13 technical indicators calculate hote hain:
    MA10, MA20, MA50, EMA12, EMA26, MACD, RSI, Bollinger Band Width, Volume Ratio, Price Change, High-Low%, Close, Volume.<br/>
    <b>Step 3 — Model Training:</b><br/>
    &nbsp;&nbsp;• <b>Random Forest Classifier</b> — next day price UP ya DOWN predict karta hai (80/20 train-test split).<br/>
    &nbsp;&nbsp;• <b>Linear Regression</b> — next day exact closing price predict karta hai.<br/>
    <b>Step 4 — Prediction:</b> Latest market data pe model apply hota hai → direction + confidence + price target milta hai.<br/>
    <b>Step 5 — Output:</b> Dashboard pe /api/predict route se JSON response aata hai jisme direction, confidence aur predicted price hota hai.<br/>
    <b>Step 6 — Model Save:</b> Trained model <i>models/</i> folder mein .pkl file ke roop mein save hota hai — baar baar retrain nahi karna padta.
    """
    story.append(Paragraph(how_text, body_style))
    story.append(Spacer(1, 0.3*cm))

    # Actual vs Predicted Chart
    story.append(Paragraph("4. Actual vs Predicted Price (Test Set)", h2_style))
    buf1 = _make_chart_actual_vs_pred(avp, symbol)
    story.append(RLImage(buf1, width=16*cm, height=5.5*cm))
    story.append(Spacer(1, 0.3*cm))

    # Feature Importance Chart
    story.append(Paragraph("5. Feature Importance", h2_style))
    buf2 = _make_chart_feature_importance(feat_importance, symbol)
    story.append(RLImage(buf2, width=14*cm, height=5.5*cm))
    story.append(Spacer(1, 0.3*cm))

    # Precision Matrix
    story.append(Paragraph("6. Precision Matrix (Direction Classifier)", h2_style))
    buf3 = _make_chart_confusion(clf_report, symbol)
    story.append(RLImage(buf3, width=8*cm, height=6*cm))
    story.append(Spacer(1, 0.3*cm))

    # API Usage
    story.append(Paragraph("7. API Endpoints", h2_style))
    api_data = [
        ['Endpoint', 'Method', 'Description'],
        ['/api/predict', 'POST', 'Predict next day direction + price for a symbol'],
        ['/api/train_model', 'POST', 'Train/retrain model for a symbol'],
        ['/api/model_report', 'POST', 'Generate & download PDF report for a symbol'],
        ['/api/model_status', 'POST', 'Check if model exists + last trained time'],
    ]
    t3 = Table(api_data, colWidths=[5*cm, 2.5*cm, 9.5*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.5*cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        f"Generated by Stock Tracker AI &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y')} &nbsp;|&nbsp; Model: RandomForest + LinearRegression",
        ParagraphStyle('footer', fontSize=8, textColor=colors.HexColor('#999999'), alignment=TA_CENTER, spaceBefore=6)
    ))

    doc.build(story)
    return filepath, filename
