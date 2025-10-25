"""
PDF Report Generator untuk Model Comparison: Linear Regression vs MLP Backpropagation
Generates comprehensive PDF report with analysis and results
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

class ModelComparisonPDFReport:
    """
    Kelas untuk membuat laporan PDF comparison model
    """
    
    def __init__(self, output_filename="Model_Comparison_Report.pdf"):
        self.filename = output_filename
        self.doc = SimpleDocTemplate(
            self.filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Define custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#2E4057')
        )
        
        self.heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=HexColor('#2E4057'),
            borderWidth=1,
            borderColor=HexColor('#2E4057'),
            borderPadding=5
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Hasil dari comparison (hardcoded berdasarkan output Anda)
        self.results = {
            'linear_regression': {
                'training_time': 0.0076,
                'test_r2': 0.6144,
                'test_rmse': 71084.13,
                'test_mae': 51835.73
            },
            'mlp_backpropagation': {
                'training_time': 40.1458,
                'test_r2': 0.6721,
                'test_rmse': 65553.13,
                'test_mae': 46228.38,
                'iterations': 443
            }
        }
        
    def add_title_page(self):
        """Membuat halaman judul"""
        # Logo/Header space
        self.story.append(Spacer(1, 0.5*inch))
        
        # Main Title
        title = Paragraph(
            "LAPORAN PERBANDINGAN MODEL<br/>LINEAR REGRESSION vs MLP BACKPROPAGATION", 
            self.title_style
        )
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        subtitle = Paragraph(
            "Analisis Prediksi Harga Rumah menggunakan California Housing Dataset", 
            ParagraphStyle('subtitle', parent=self.styles['Normal'], 
                          fontSize=12, alignment=TA_CENTER, 
                          textColor=HexColor('#666666'))
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Project Info Box
        project_info = [
            ['Project', 'Deep Learning Model Comparison'],
            ['Dataset', 'California Housing Dataset'],
            ['Models', 'Linear Regression & MLP Backpropagation'],
            ['Samples', '20,640 data points'],
            ['Features', '8 numeric features'],
            ['Target', 'median_house_value'],
            ['Date', datetime.now().strftime('%B %d, %Y')]
        ]
        
        info_table = Table(project_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E9ECEF')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        
        self.story.append(info_table)
        self.story.append(Spacer(1, 0.8*inch))
        
        # Executive Summary Box
        summary_text = """
        <b>EXECUTIVE SUMMARY</b><br/><br/>
        Penelitian ini membandingkan performa Linear Regression dengan MLP Backpropagation 
        untuk prediksi harga rumah. Hasil menunjukkan bahwa MLP Backpropagation memberikan 
        peningkatan akurasi sebesar 9.4% (R² improvement) dan pengurangan error sebesar 7.8% 
        (RMSE reduction) dibandingkan Linear Regression, dengan trade-off training time 
        yang 5,283x lebih lama.
        """
        
        summary = Paragraph(summary_text, ParagraphStyle(
            'summary', parent=self.styles['Normal'], 
            fontSize=10, alignment=TA_JUSTIFY,
            borderWidth=2, borderColor=HexColor('#28A745'),
            borderPadding=15, backColor=HexColor('#F8FFF8')
        ))
        self.story.append(summary)
        
        self.story.append(PageBreak())
        
    def add_methodology_section(self):
        """Menambahkan section metodologi"""
        # Section Title
        title = Paragraph("1. METODOLOGI PENELITIAN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Dataset Description
        dataset_text = """
        <b>1.1 Dataset California Housing</b><br/>
        Dataset yang digunakan adalah California Housing dataset yang berisi informasi 
        perumahan di California. Dataset ini terdiri dari 20,640 sampel dengan 8 fitur 
        numerik dan 1 target variabel (median_house_value).
        """
        self.story.append(Paragraph(dataset_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Features table
        features_data = [
            ['Feature', 'Description', 'Type'],
            ['longitude', 'Koordinat longitude lokasi', 'Numeric'],
            ['latitude', 'Koordinat latitude lokasi', 'Numeric'],
            ['housing_median_age', 'Median umur rumah dalam blok', 'Numeric'],
            ['total_rooms', 'Total jumlah kamar dalam blok', 'Numeric'],
            ['total_bedrooms', 'Total jumlah kamar tidur dalam blok', 'Numeric'],
            ['population', 'Total populasi dalam blok', 'Numeric'],
            ['households', 'Total rumah tangga dalam blok', 'Numeric'],
            ['median_income', 'Median pendapatan dalam blok', 'Numeric']
        ]
        
        features_table = Table(features_data, colWidths=[1.5*inch, 2.5*inch, 1*inch])
        features_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E9ECEF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(features_table)
        self.story.append(Spacer(1, 12))
        
        # Preprocessing
        preprocessing_text = """
        <b>1.2 Preprocessing Data</b><br/>
        • <b>Missing Value Handling:</b> Menggunakan SimpleImputer dengan strategi mean untuk mengatasi nilai yang hilang<br/>
        • <b>Feature Scaling:</b> StandardScaler diterapkan untuk MLP, tidak untuk Linear Regression<br/>
        • <b>Train-Test Split:</b> 80% untuk training (16,512 samples) dan 20% untuk testing (4,128 samples)<br/>
        • <b>Random State:</b> 42 untuk reproducibility
        """
        self.story.append(Paragraph(preprocessing_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # Model Descriptions
        models_text = """
        <b>1.3 Model Descriptions</b><br/><br/>
        <b>Linear Regression:</b><br/>
        • Algorithm: Ordinary Least Squares (OLS)<br/>
        • Assumptions: Linear relationship antara features dan target<br/>
        • No regularization<br/>
        • Scikit-learn implementation<br/><br/>
        
        <b>MLP Backpropagation:</b><br/>
        • Architecture: Input(8) → Hidden(100, 50) → Output(1)<br/>
        • Activation Function: ReLU<br/>
        • Optimizer: Adam dengan learning rate 0.001<br/>
        • Regularization: L2 dengan alpha=0.001<br/>
        • Early Stopping: Ya, dengan validation_fraction=0.1<br/>
        • Max Iterations: 500 dengan n_iter_no_change=10
        """
        self.story.append(Paragraph(models_text, self.normal_style))
        
        self.story.append(PageBreak())
        
    def add_results_section(self):
        """Menambahkan section hasil"""
        title = Paragraph("2. HASIL EKSPERIMEN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Results Summary
        results_text = """
        <b>2.1 Performance Comparison</b><br/>
        Berikut adalah hasil perbandingan performa kedua model pada test dataset:
        """
        self.story.append(Paragraph(results_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Results table
        results_data = [
            ['Metric', 'Linear Regression', 'MLP Backpropagation', 'Improvement'],
            ['Training Time (seconds)', f"{self.results['linear_regression']['training_time']:.4f}", 
             f"{self.results['mlp_backpropagation']['training_time']:.4f}", 
             f"{self.results['mlp_backpropagation']['training_time']/self.results['linear_regression']['training_time']:.0f}x slower"],
            ['Test R² Score', f"{self.results['linear_regression']['test_r2']:.4f}", 
             f"{self.results['mlp_backpropagation']['test_r2']:.4f}", 
             f"{((self.results['mlp_backpropagation']['test_r2']/self.results['linear_regression']['test_r2'])-1)*100:.1f}%"],
            ['Test RMSE', f"{self.results['linear_regression']['test_rmse']:.2f}", 
             f"{self.results['mlp_backpropagation']['test_rmse']:.2f}", 
             f"{((self.results['linear_regression']['test_rmse']/self.results['mlp_backpropagation']['test_rmse'])-1)*100:.1f}%"],
            ['Test MAE', f"{self.results['linear_regression']['test_mae']:.2f}", 
             f"{self.results['mlp_backpropagation']['test_mae']:.2f}", 
             f"{((self.results['linear_regression']['test_mae']/self.results['mlp_backpropagation']['test_mae'])-1)*100:.1f}%"],
            ['Iterations', 'N/A', f"{self.results['mlp_backpropagation']['iterations']}", 'N/A']
        ]
        
        results_table = Table(results_data, colWidths=[1.3*inch, 1.3*inch, 1.5*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E9ECEF')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            # Highlight improvement column
            ('BACKGROUND', (3, 1), (3, -1), HexColor('#E8F5E8')),
            ('TEXTCOLOR', (3, 2), (3, 4), HexColor('#28A745'))  # Green for positive improvements
        ]))
        
        self.story.append(results_table)
        self.story.append(Spacer(1, 12))
        
        # Key Findings
        findings_text = """
        <b>2.2 Key Findings</b><br/>
        • <b>Accuracy Improvement:</b> MLP Backpropagation mencapai R² score 0.6721 vs 0.6144 untuk Linear Regression (9.4% improvement)<br/>
        • <b>Error Reduction:</b> RMSE berkurang dari 71,084 menjadi 65,553 (7.8% reduction)<br/>
        • <b>MAE Improvement:</b> Mean Absolute Error berkurang dari 51,836 menjadi 46,228 (10.8% reduction)<br/>
        • <b>Training Cost:</b> MLP membutuhkan waktu 5,283x lebih lama (40.15s vs 0.0076s)<br/>
        • <b>Convergence:</b> MLP converged dalam 443 iterations dari maksimal 500
        """
        self.story.append(Paragraph(findings_text, self.normal_style))
        
        self.story.append(PageBreak())
        
    def add_analysis_section(self):
        """Menambahkan section analisis"""
        title = Paragraph("3. ANALISIS DAN PEMBAHASAN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Performance Analysis
        perf_text = """
        <b>3.1 Analisis Performa</b><br/>
        MLP Backpropagation menunjukkan performa yang superior dibandingkan Linear Regression 
        dalam semua metrics evaluasi. Peningkatan R² score sebesar 9.4% mengindikasikan bahwa 
        MLP mampu menangkap non-linear relationships dalam data yang tidak dapat dideteksi 
        oleh Linear Regression.
        """
        self.story.append(Paragraph(perf_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Computational Cost Analysis
        cost_text = """
        <b>3.2 Analisis Computational Cost</b><br/>
        Trade-off utama adalah computational cost. MLP membutuhkan waktu training 5,283x 
        lebih lama dibandingkan Linear Regression. Untuk dataset dengan 16,512 training samples, 
        Linear Regression hanya membutuhkan 7.6ms sedangkan MLP membutuhkan 40.15 detik.
        """
        self.story.append(Paragraph(cost_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Model Complexity Analysis
        complexity_text = """
        <b>3.3 Analisis Kompleksitas Model</b><br/>
        • <b>Linear Regression:</b> Model sederhana dengan 8 parameters (weights) + 1 bias = 9 parameters total<br/>
        • <b>MLP Backpropagation:</b> Model kompleks dengan (8×100) + (100×50) + (50×1) + biases = 5,951 parameters total<br/>
        • <b>Interpretability:</b> Linear Regression lebih mudah diinterpretasi dengan koefisien yang jelas<br/>
        • <b>Generalization:</b> Kedua model menunjukkan generalization yang baik tanpa overfitting signifikan
        """
        self.story.append(Paragraph(complexity_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # When to Use Each Model
        usage_text = """
        <b>3.4 Rekomendasi Penggunaan</b><br/><br/>
        <b>Gunakan Linear Regression ketika:</b><br/>
        • Real-time predictions diperlukan<br/>
        • Interpretability model penting<br/>
        • Computational resources terbatas<br/>
        • Baseline model untuk comparison<br/>
        • Linear relationships cukup untuk problem domain<br/><br/>
        
        <b>Gunakan MLP Backpropagation ketika:</b><br/>
        • Accuracy adalah prioritas utama<br/>
        • Complex non-linear patterns perlu ditangkap<br/>
        • Training time bukan constraint utama<br/>
        • Sufficient computational resources tersedia<br/>
        • Improvement 7-10% worth the extra complexity
        """
        self.story.append(Paragraph(usage_text, self.normal_style))
        
        self.story.append(PageBreak())
        
    def add_conclusion_section(self):
        """Menambahkan section kesimpulan"""
        title = Paragraph("4. KESIMPULAN DAN SARAN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Conclusion
        conclusion_text = """
        <b>4.1 Kesimpulan</b><br/>
        Berdasarkan eksperimen yang telah dilakukan, dapat disimpulkan bahwa:<br/><br/>
        
        1. <b>MLP Backpropagation</b> memberikan performa yang superior dengan peningkatan 
           R² score sebesar 9.4% dan pengurangan RMSE sebesar 7.8% dibandingkan Linear Regression.<br/><br/>
        
        2. <b>Trade-off computational cost</b> sangat signifikan, dengan MLP membutuhkan waktu 
           training 5,283x lebih lama, yang menjadi pertimbangan penting dalam implementasi.<br/><br/>
        
        3. <b>Dataset California Housing</b> mengandung non-linear patterns yang dapat 
           dieksploitasi oleh MLP untuk memberikan prediksi yang lebih akurat.<br/><br/>
        
        4. <b>Kedua model</b> menunjukkan generalization yang baik tanpa overfitting yang signifikan.<br/><br/>
        
        5. <b>Pilihan model</b> sangat bergantung pada requirements spesifik: speed vs accuracy.
        """
        self.story.append(Paragraph(conclusion_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # Recommendations
        recommendations_text = """
        <b>4.2 Saran untuk Penelitian Selanjutnya</b><br/>
        • Eksplorasi arsitektur MLP yang berbeda untuk optimasi lebih lanjut<br/>
        • Implementasi regularization techniques untuk mencegah overfitting<br/>
        • Perbandingan dengan algoritma ML lainnya (Random Forest, XGBoost, etc.)<br/>
        • Analisis feature importance untuk understanding model behavior<br/>
        • Hyperparameter tuning untuk optimasi performa MLP<br/>
        • Cross-validation untuk validasi hasil yang lebih robust<br/>
        • Implementasi ensemble methods untuk kombinasi model strengths
        """
        self.story.append(Paragraph(recommendations_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # Final Recommendation Box
        final_rec = """
        <b>REKOMENDASI FINAL</b><br/><br/>
        Untuk implementasi production pada California Housing prediction:<br/><br/>
        • <b>High-accuracy requirements:</b> Gunakan MLP Backpropagation<br/>
        • <b>Real-time/fast predictions:</b> Gunakan Linear Regression<br/>
        • <b>Balanced approach:</b> Pertimbangkan ensemble atau hybrid approach
        """
        
        final_box = Paragraph(final_rec, ParagraphStyle(
            'final', parent=self.styles['Normal'], 
            fontSize=10, alignment=TA_JUSTIFY,
            borderWidth=2, borderColor=HexColor('#007BFF'),
            borderPadding=15, backColor=HexColor('#F0F8FF')
        ))
        self.story.append(final_box)
        
    def generate_chart(self):
        """Generate comparison chart"""
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Comparison: Linear Regression vs MLP Backpropagation', 
                     fontsize=14, fontweight='bold')
        
        models = ['Linear Regression', 'MLP Backpropagation']
        
        # R² Comparison
        r2_scores = [self.results['linear_regression']['test_r2'], 
                    self.results['mlp_backpropagation']['test_r2']]
        bars1 = ax1.bar(models, r2_scores, color=['skyblue', 'lightcoral'])
        ax1.set_title('Test R² Score Comparison')
        ax1.set_ylabel('R² Score')
        for i, v in enumerate(r2_scores):
            ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # RMSE Comparison
        rmse_scores = [self.results['linear_regression']['test_rmse'], 
                      self.results['mlp_backpropagation']['test_rmse']]
        bars2 = ax2.bar(models, rmse_scores, color=['skyblue', 'lightcoral'])
        ax2.set_title('Test RMSE Comparison')
        ax2.set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            ax2.text(i, v + 1000, f'{v:.0f}', ha='center', va='bottom')
        
        # Training Time Comparison
        times = [self.results['linear_regression']['training_time'], 
                self.results['mlp_backpropagation']['training_time']]
        bars3 = ax3.bar(models, times, color=['skyblue', 'lightcoral'])
        ax3.set_title('Training Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_yscale('log')  # Log scale due to huge difference
        for i, v in enumerate(times):
            ax3.text(i, v * 1.5, f'{v:.4f}s', ha='center', va='bottom')
        
        # MAE Comparison
        mae_scores = [self.results['linear_regression']['test_mae'], 
                     self.results['mlp_backpropagation']['test_mae']]
        bars4 = ax4.bar(models, mae_scores, color=['skyblue', 'lightcoral'])
        ax4.set_title('Test MAE Comparison')
        ax4.set_ylabel('MAE')
        for i, v in enumerate(mae_scores):
            ax4.text(i, v + 1000, f'{v:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = r'c:\deep learning\comparison_chart.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    def generate_report(self):
        """Generate complete PDF report"""
        print("🚀 Generating PDF Report...")
        
        # Add all sections
        self.add_title_page()
        self.add_methodology_section()
        self.add_results_section()
        self.add_analysis_section()
        self.add_conclusion_section()
        
        # Generate chart and add to report
        chart_path = self.generate_chart()
        
        # Add chart page
        self.story.append(PageBreak())
        chart_title = Paragraph("5. VISUALISASI HASIL", self.heading2_style)
        self.story.append(chart_title)
        self.story.append(Spacer(1, 12))
        
        try:
            chart_img = Image(chart_path, width=7*inch, height=5.8*inch)
            self.story.append(chart_img)
        except:
            chart_text = Paragraph("Chart visualization tersimpan sebagai comparison_chart.png", 
                                 self.normal_style)
            self.story.append(chart_text)
        
        # Build PDF
        self.doc.build(self.story)
        
        print(f"✅ PDF Report berhasil dibuat: {self.filename}")
        print(f"✅ Comparison chart tersimpan: {chart_path}")
        
        # Clean up chart file
        if os.path.exists(chart_path):
            print(f"📊 Chart file: {chart_path}")

def main():
    """Main function untuk generate PDF report"""
    try:
        # Generate report
        report_path = r"c:\deep learning\Model_Comparison_Report.pdf"
        reporter = ModelComparisonPDFReport(report_path)
        reporter.generate_report()
        
        print("\n" + "="*60)
        print("🎉 PDF REPORT GENERATION COMPLETED!")
        print("="*60)
        print(f"📄 Report file: {report_path}")
        print("📊 Chart file: comparison_chart.png")
        print("\n📋 Report includes:")
        print("   ✓ Executive Summary")
        print("   ✓ Methodology & Dataset Description")
        print("   ✓ Detailed Results Analysis")
        print("   ✓ Performance Comparison Tables")
        print("   ✓ Visualization Charts")
        print("   ✓ Conclusions & Recommendations")
        
    except ImportError as e:
        print("❌ Error: ReportLab tidak terinstall.")
        print("💡 Install dengan: pip install reportlab")
        print(f"   Detail error: {e}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

if __name__ == "__main__":
    main()