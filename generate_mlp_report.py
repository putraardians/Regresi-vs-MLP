"""
PDF Report Generator untuk MLP Architecture Comparison
Generates comprehensive PDF report untuk hasil eksperimen MLP dengan berbagai arsitektur
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

class MLPArchitectureComparisonReport:
    """
    Kelas untuk membuat laporan PDF comparison arsitektur MLP
    """
    
    def __init__(self, output_filename="MLP_Architecture_Comparison_Report.pdf"):
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
            textColor=HexColor('#1F4E79')
        )
        
        self.heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=HexColor('#1F4E79'),
            borderWidth=1,
            borderColor=HexColor('#1F4E79'),
            borderPadding=5
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Hasil eksperimen berdasarkan output yang diberikan
        self.results = {
            'Percobaan 1': {
                'architecture': (16,),
                'time_s': 34.06,
                'n_iter': 1647,
                'hit_max_iter': False,
                'convergence_warnings': 0,
                'mae': 55890.87686634171,
                'mse': 5859245905.193248,
                'rmse': 76545.71121358301
            },
            'Percobaan 2': {
                'architecture': (16, 16),
                'time_s': 13.988317489624023,
                'n_iter': 461,
                'hit_max_iter': False,
                'convergence_warnings': 0,
                'mae': 55158.608822188085,
                'mse': 5631533633.23186,
                'rmse': 75043.54491381561
            },
            'Percobaan 3': {
                'architecture': (16, 16, 32),
                'time_s': 20.133867979049683,
                'n_iter': 406,
                'hit_max_iter': False,
                'convergence_warnings': 0,
                'mae': 54199.445587949434,
                'mse': 5433621217.197945,
                'rmse': 73713.10071620882
            }
        }
        
    def add_title_page(self):
        """Membuat halaman judul"""
        # Logo/Header space
        self.story.append(Spacer(1, 0.5*inch))
        
        # Main Title
        title = Paragraph(
            "LAPORAN ANALISIS PERBANDINGAN<br/>ARSITEKTUR MLP BACKPROPAGATION", 
            self.title_style
        )
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        subtitle = Paragraph(
            "Eksperimen Multi-Layer Perceptron dengan Berbagai Hidden Layer Configurations<br/>untuk Prediksi Harga Rumah California Housing Dataset", 
            ParagraphStyle('subtitle', parent=self.styles['Normal'], 
                          fontSize=12, alignment=TA_CENTER, 
                          textColor=HexColor('#666666'))
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Project Info Box
        project_info = [
            ['Experiment Type', 'MLP Architecture Comparison'],
            ['Dataset', 'California Housing Dataset'],
            ['Algorithm', 'Multi-Layer Perceptron (MLPRegressor)'],
            ['Activation Function', 'ReLU'],
            ['Solver', 'Adam Optimizer'],
            ['Max Iterations', '2000'],
            ['Early Stopping', 'Enabled'],
            ['Architecture Tested', '3 configurations'],
            ['Date', datetime.now().strftime('%B %d, %Y')]
        ]
        
        info_table = Table(project_info, colWidths=[2.2*inch, 2.8*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#F0F8FF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#B0C4DE')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        
        self.story.append(info_table)
        self.story.append(Spacer(1, 0.8*inch))
        
        # Executive Summary Box
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_rmse = self.results[best_model]['rmse']
        worst_model = max(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        worst_rmse = self.results[worst_model]['rmse']
        improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
        
        summary_text = f"""
        <b>EXECUTIVE SUMMARY</b><br/><br/>
        Eksperimen ini mengevaluasi tiga arsitektur MLP yang berbeda untuk prediksi harga rumah. 
        Hasil menunjukkan bahwa <b>{best_model}</b> dengan arsitektur {self.results[best_model]['architecture']} 
        memberikan performa terbaik dengan RMSE {best_rmse:.2f}, menghasilkan improvement sebesar 
        {improvement:.2f}% dibandingkan arsitektur terburuk. Semua model berhasil converge tanpa 
        mencapai maximum iterations, menunjukkan stabilitas training yang baik.
        """
        
        summary = Paragraph(summary_text, ParagraphStyle(
            'summary', parent=self.styles['Normal'], 
            fontSize=10, alignment=TA_JUSTIFY,
            borderWidth=2, borderColor=HexColor('#4682B4'),
            borderPadding=15, backColor=HexColor('#F0F8FF')
        ))
        self.story.append(summary)
        
        self.story.append(PageBreak())
        
    def add_methodology_section(self):
        """Menambahkan section metodologi"""
        # Section Title
        title = Paragraph("1. METODOLOGI EKSPERIMEN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Experimental Setup
        setup_text = """
        <b>1.1 Setup Eksperimen</b><br/>
        Eksperimen ini menggunakan scikit-learn MLPRegressor untuk mengevaluasi performa 
        berbagai arsitektur neural network dalam memprediksi harga rumah. Dataset California 
        Housing digunakan sebagai benchmark dengan 5 fitur input yang telah diseleksi.
        """
        self.story.append(Paragraph(setup_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Dataset Processing
        dataset_text = """
        <b>1.2 Dataset Processing</b><br/>
        • <b>Dataset:</b> California Housing Dataset<br/>
        • <b>Feature Selection:</b> 5 fitur numerik terpilih (median_income, total_rooms, total_bedrooms, population, households)<br/>
        • <b>Missing Value Handling:</b> Mean imputation menggunakan SimpleImputer<br/>
        • <b>Train-Test Split:</b> 80%-20% dengan random_state=1<br/>
        • <b>Target Variable:</b> median_house_value
        """
        self.story.append(Paragraph(dataset_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Model Configuration
        config_text = """
        <b>1.3 Model Configuration</b><br/>
        Semua model menggunakan konfigurasi yang sama kecuali hidden layer architecture:
        """
        self.story.append(Paragraph(config_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Configuration table
        config_data = [
            ['Parameter', 'Value', 'Description'],
            ['Activation Function', 'ReLU', 'Rectified Linear Unit activation'],
            ['Solver', 'Adam', 'Adaptive moment estimation optimizer'],
            ['Random State', '1', 'For reproducibility'],
            ['Max Iterations', '2000', 'Maximum training iterations'],
            ['Early Stopping', 'True', 'Stop when no improvement'],
            ['N Iter No Change', '20', 'Patience for early stopping'],
            ['Alpha', 'Default (0.0001)', 'L2 regularization parameter']
        ]
        
        config_table = Table(config_data, colWidths=[1.5*inch, 1.2*inch, 2.3*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1F4E79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E9ECEF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(config_table)
        self.story.append(Spacer(1, 12))
        
        # Architecture Descriptions
        arch_text = """
        <b>1.4 Arsitektur yang Diuji</b><br/>
        Tiga arsitektur MLP yang berbeda diuji untuk mengevaluasi pengaruh kompleksitas 
        terhadap performa:
        """
        self.story.append(Paragraph(arch_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        arch_data = [
            ['Percobaan', 'Architecture', 'Total Layers', 'Parameters*', 'Complexity'],
            ['Percobaan 1', '(16,)', '3 layers', '~113', 'Simple'],
            ['Percobaan 2', '(16, 16)', '4 layers', '~369', 'Medium'],
            ['Percobaan 3', '(16, 16, 32)', '5 layers', '~913', 'Complex']
        ]
        
        arch_table = Table(arch_data, colWidths=[1*inch, 1.2*inch, 1*inch, 1*inch, 0.8*inch])
        arch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4682B4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F0F8FF')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#B0C4DE')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(arch_table)
        self.story.append(Spacer(1, 6))
        
        note_text = """
        <i>*Parameter count approximate: Input(5) → Hidden layers → Output(1)</i>
        """
        self.story.append(Paragraph(note_text, ParagraphStyle(
            'note', parent=self.styles['Normal'], fontSize=8, 
            textColor=HexColor('#666666'), alignment=TA_CENTER
        )))
        
        self.story.append(PageBreak())
        
    def add_results_section(self):
        """Menambahkan section hasil"""
        title = Paragraph("2. HASIL EKSPERIMEN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Results Overview
        results_text = """
        <b>2.1 Overview Hasil</b><br/>
        Berikut adalah hasil lengkap dari ketiga eksperimen arsitektur MLP:
        """
        self.story.append(Paragraph(results_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Main Results Table
        results_data = [
            ['Metric', 'Percobaan 1\n(16,)', 'Percobaan 2\n(16,16)', 'Percobaan 3\n(16,16,32)'],
            ['Training Time (s)', f"{self.results['Percobaan 1']['time_s']:.2f}", 
             f"{self.results['Percobaan 2']['time_s']:.2f}", 
             f"{self.results['Percobaan 3']['time_s']:.2f}"],
            ['Iterations', f"{self.results['Percobaan 1']['n_iter']}", 
             f"{self.results['Percobaan 2']['n_iter']}", 
             f"{self.results['Percobaan 3']['n_iter']}"],
            ['Converged', 'Yes', 'Yes', 'Yes'],
            ['Convergence Warnings', '0', '0', '0'],
            ['MAE', f"{self.results['Percobaan 1']['mae']:.2f}", 
             f"{self.results['Percobaan 2']['mae']:.2f}", 
             f"{self.results['Percobaan 3']['mae']:.2f}"],
            ['MSE', f"{self.results['Percobaan 1']['mse']:.0f}", 
             f"{self.results['Percobaan 2']['mse']:.0f}", 
             f"{self.results['Percobaan 3']['mse']:.0f}"],
            ['RMSE', f"{self.results['Percobaan 1']['rmse']:.2f}", 
             f"{self.results['Percobaan 2']['rmse']:.2f}", 
             f"{self.results['Percobaan 3']['rmse']:.2f}"]
        ]
        
        results_table = Table(results_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1F4E79')),
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
            # Highlight best results (Percobaan 3 - lowest RMSE)
            ('BACKGROUND', (3, 7), (3, 7), HexColor('#90EE90')),  # Best RMSE
            ('BACKGROUND', (3, 5), (3, 5), HexColor('#90EE90')),  # Best MAE
            ('BACKGROUND', (3, 6), (3, 6), HexColor('#90EE90')),  # Best MSE
        ]))
        
        self.story.append(results_table)
        self.story.append(Spacer(1, 12))
        
        # Performance Ranking
        ranking_text = """
        <b>2.2 Performance Ranking</b><br/>
        Berdasarkan RMSE (Root Mean Square Error) sebagai metric utama:
        """
        self.story.append(Paragraph(ranking_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Sort by RMSE
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['rmse'])
        
        ranking_data = [['Rank', 'Experiment', 'Architecture', 'RMSE', 'Improvement']]
        base_rmse = sorted_results[-1][1]['rmse']  # Worst RMSE for improvement calculation
        
        for i, (name, result) in enumerate(sorted_results, 1):
            improvement = ((base_rmse - result['rmse']) / base_rmse) * 100 if i > 1 else 0
            ranking_data.append([
                f"{i}",
                name,
                str(result['architecture']),
                f"{result['rmse']:.2f}",
                f"{improvement:.2f}%" if i > 1 else "Baseline"
            ])
        
        ranking_table = Table(ranking_data, colWidths=[0.6*inch, 1.2*inch, 1*inch, 1*inch, 1.2*inch])
        ranking_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4682B4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (0, 1), HexColor('#FFD700')),  # Gold for 1st place
            ('BACKGROUND', (0, 2), (0, 2), HexColor('#C0C0C0')),  # Silver for 2nd place
            ('BACKGROUND', (0, 3), (0, 3), HexColor('#CD7F32')),  # Bronze for 3rd place
            ('BACKGROUND', (0, 2), (-1, -1), HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E9ECEF')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(ranking_table)
        
        self.story.append(PageBreak())
        
    def add_analysis_section(self):
        """Menambahkan section analisis"""
        title = Paragraph("3. ANALISIS DAN PEMBAHASAN", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        # Performance Analysis
        best_exp = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        worst_exp = max(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        improvement = ((self.results[worst_exp]['rmse'] - self.results[best_exp]['rmse']) / self.results[worst_exp]['rmse']) * 100
        
        perf_text = f"""
        <b>3.1 Analisis Performa</b><br/>
        <b>{best_exp}</b> dengan arsitektur {self.results[best_exp]['architecture']} menunjukkan 
        performa terbaik dengan RMSE {self.results[best_exp]['rmse']:.2f}, memberikan improvement 
        sebesar {improvement:.2f}% dibandingkan {worst_exp}. Trend menunjukkan bahwa arsitektur 
        yang lebih kompleks cenderung memberikan performa yang lebih baik, namun dengan trade-off 
        computational cost yang berbeda.
        """
        self.story.append(Paragraph(perf_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Training Efficiency Analysis
        fastest_exp = min(self.results.keys(), key=lambda x: self.results[x]['time_s'])
        slowest_exp = max(self.results.keys(), key=lambda x: self.results[x]['time_s'])
        
        efficiency_text = f"""
        <b>3.2 Analisis Efisiensi Training</b><br/>
        <b>{fastest_exp}</b> menunjukkan training time tercepat ({self.results[fastest_exp]['time_s']:.2f}s) 
        sedangkan <b>{slowest_exp}</b> membutuhkan waktu terlama ({self.results[slowest_exp]['time_s']:.2f}s). 
        Menariknya, model tercepat bukan yang paling sederhana, menunjukkan bahwa arsitektur yang 
        tepat dapat mencapai convergence lebih efisien.
        """
        self.story.append(Paragraph(efficiency_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Convergence Analysis
        convergence_text = """
        <b>3.3 Analisis Convergence</b><br/>
        Semua model berhasil converge tanpa mencapai maximum iterations (2000), menunjukkan:
        """
        self.story.append(Paragraph(convergence_text, self.normal_style))
        self.story.append(Spacer(1, 6))
        
        # Convergence details
        conv_data = [
            ['Experiment', 'Iterations Used', 'Max Iterations', 'Convergence %', 'Status'],
            ['Percobaan 1', f"{self.results['Percobaan 1']['n_iter']}", '2000', 
             f"{(self.results['Percobaan 1']['n_iter']/2000)*100:.1f}%", '✓ Converged'],
            ['Percobaan 2', f"{self.results['Percobaan 2']['n_iter']}", '2000', 
             f"{(self.results['Percobaan 2']['n_iter']/2000)*100:.1f}%", '✓ Converged'],
            ['Percobaan 3', f"{self.results['Percobaan 3']['n_iter']}", '2000', 
             f"{(self.results['Percobaan 3']['n_iter']/2000)*100:.1f}%", '✓ Converged']
        ]
        
        conv_table = Table(conv_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
        conv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F0FFF0')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#90EE90')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        
        self.story.append(conv_table)
        self.story.append(Spacer(1, 12))
        
        # Key Insights
        insights_text = """
        <b>3.4 Key Insights</b><br/>
        • <b>Architecture Complexity:</b> Peningkatan kompleksitas arsitektur memberikan improvement yang konsisten<br/>
        • <b>Training Stability:</b> Semua model menunjukkan training yang stabil tanpa convergence issues<br/>
        • <b>Efficiency Trade-off:</b> Percobaan 2 memberikan balance terbaik antara speed dan accuracy<br/>
        • <b>Scalability:</b> Penambahan layer dan neurons memberikan diminishing returns yang reasonable<br/>
        • <b>Generalization:</b> Tidak ada indikasi overfitting atau instability dalam training process
        """
        self.story.append(Paragraph(insights_text, self.normal_style))
        
        self.story.append(PageBreak())
        
    def add_conclusion_section(self):
        """Menambahkan section kesimpulan"""
        title = Paragraph("4. KESIMPULAN DAN REKOMENDASI", self.heading2_style)
        self.story.append(title)
        self.story.append(Spacer(1, 12))
        
        best_exp = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_result = self.results[best_exp]
        improvement = ((self.results['Percobaan 1']['rmse'] - best_result['rmse']) / self.results['Percobaan 1']['rmse']) * 100
        
        # Main Conclusions
        conclusion_text = f"""
        <b>4.1 Kesimpulan Utama</b><br/>
        Berdasarkan eksperimen yang telah dilakukan, dapat disimpulkan bahwa:<br/><br/>
        
        1. <b>Arsitektur Optimal:</b> {best_exp} dengan konfigurasi {best_result['architecture']} 
           memberikan performa terbaik dengan RMSE {best_result['rmse']:.2f}.<br/><br/>
        
        2. <b>Performance Improvement:</b> Arsitektur terbaik memberikan improvement sebesar 
           {improvement:.2f}% dibandingkan arsitektur paling sederhana.<br/><br/>
        
        3. <b>Training Stability:</b> Semua arsitektur menunjukkan convergence yang baik tanpa 
           mencapai maximum iterations, menandakan konfigurasi hyperparameter yang tepat.<br/><br/>
        
        4. <b>Complexity vs Performance:</b> Terdapat correlation positif antara complexity 
           arsitektur dengan performance, namun dengan diminishing returns.<br/><br/>
        
        5. <b>Computational Efficiency:</b> Percobaan 2 (16,16) menunjukkan balance terbaik 
           antara training time dan accuracy.
        """
        self.story.append(Paragraph(conclusion_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # Recommendations
        recommendations_text = """
        <b>4.2 Rekomendasi</b><br/><br/>
        <b>Untuk Production Implementation:</b><br/>
        • <b>Best Performance:</b> Gunakan arsitektur (16, 16, 32) untuk maximum accuracy<br/>
        • <b>Balanced Approach:</b> Pertimbangkan arsitektur (16, 16) untuk balance speed-accuracy<br/>
        • <b>Resource Constrained:</b> Arsitektur (16,) tetap viable dengan performance yang acceptable<br/><br/>
        
        <b>Untuk Further Research:</b><br/>
        • Eksplorasi arsitektur yang lebih dalam dengan regularization techniques<br/>
        • Hyperparameter tuning untuk learning rate, alpha, dan batch size<br/>
        • Cross-validation untuk validasi yang lebih robust<br/>
        • Comparison dengan algoritma ensemble methods<br/>
        • Feature engineering untuk improve overall performance
        """
        self.story.append(Paragraph(recommendations_text, self.normal_style))
        self.story.append(Spacer(1, 12))
        
        # Final Recommendation Box
        final_rec = f"""
        <b>REKOMENDASI FINAL</b><br/><br/>
        Untuk implementasi prediksi harga rumah California Housing:<br/><br/>
        • <b>Production Ready:</b> Gunakan {best_exp} - {best_result['architecture']}<br/>
        • <b>RMSE Target:</b> {best_result['rmse']:.2f}<br/>
        • <b>Training Time:</b> ~{best_result['time_s']:.1f} seconds<br/>
        • <b>Reliability:</b> Converged dalam {best_result['n_iter']} iterations<br/>
        • <b>Confidence Level:</b> High (no convergence warnings)
        """
        
        final_box = Paragraph(final_rec, ParagraphStyle(
            'final', parent=self.styles['Normal'], 
            fontSize=10, alignment=TA_JUSTIFY,
            borderWidth=2, borderColor=HexColor('#1F4E79'),
            borderPadding=15, backColor=HexColor('#F0F8FF')
        ))
        self.story.append(final_box)
        
    def generate_chart(self):
        """Generate comparison chart"""
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MLP Architecture Comparison Results', fontsize=16, fontweight='bold')
        
        experiments = list(self.results.keys())
        architectures = [str(self.results[exp]['architecture']) for exp in experiments]
        
        # RMSE Comparison
        rmse_values = [self.results[exp]['rmse'] for exp in experiments]
        colors_list = ['lightcoral', 'lightblue', 'lightgreen']
        bars1 = ax1.bar(architectures, rmse_values, color=colors_list)
        ax1.set_title('RMSE Comparison')
        ax1.set_ylabel('RMSE')
        ax1.set_xlabel('Architecture')
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + 500, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Training Time Comparison
        time_values = [self.results[exp]['time_s'] for exp in experiments]
        bars2 = ax2.bar(architectures, time_values, color=colors_list)
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xlabel('Architecture')
        for i, v in enumerate(time_values):
            ax2.text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Iterations Comparison
        iter_values = [self.results[exp]['n_iter'] for exp in experiments]
        bars3 = ax3.bar(architectures, iter_values, color=colors_list)
        ax3.set_title('Convergence Iterations')
        ax3.set_ylabel('Iterations')
        ax3.set_xlabel('Architecture')
        for i, v in enumerate(iter_values):
            ax3.text(i, v + 20, f'{v}', ha='center', va='bottom', fontweight='bold')
        
        # MAE Comparison
        mae_values = [self.results[exp]['mae'] for exp in experiments]
        bars4 = ax4.bar(architectures, mae_values, color=colors_list)
        ax4.set_title('MAE Comparison')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_xlabel('Architecture')
        for i, v in enumerate(mae_values):
            ax4.text(i, v + 200, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = r'c:\deep learning\mlp_architecture_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    def generate_report(self):
        """Generate complete PDF report"""
        print("🚀 Generating MLP Architecture Comparison PDF Report...")
        
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
            chart_img = Image(chart_path, width=7*inch, height=5.5*inch)
            self.story.append(chart_img)
        except:
            chart_text = Paragraph("Chart visualization tersimpan sebagai mlp_architecture_comparison.png", 
                                 self.normal_style)
            self.story.append(chart_text)
        
        # Build PDF
        self.doc.build(self.story)
        
        print(f"✅ PDF Report berhasil dibuat: {self.filename}")
        print(f"✅ Comparison chart tersimpan: {chart_path}")

def main():
    """Main function untuk generate PDF report"""
    try:
        # Generate report
        report_path = r"c:\deep learning\MLP_Architecture_Comparison_Report.pdf"
        reporter = MLPArchitectureComparisonReport(report_path)
        reporter.generate_report()
        
        print("\n" + "="*70)
        print("🎉 MLP ARCHITECTURE COMPARISON REPORT COMPLETED!")
        print("="*70)
        print(f"📄 Report file: {report_path}")
        print("📊 Chart file: mlp_architecture_comparison.png")
        print("\n📋 Report includes:")
        print("   ✓ Executive Summary dengan best model identification")
        print("   ✓ Detailed Methodology dan experimental setup")
        print("   ✓ Comprehensive Results Analysis")
        print("   ✓ Performance Ranking dan comparison tables")
        print("   ✓ Training efficiency dan convergence analysis")
        print("   ✓ Visualization charts untuk semua metrics")
        print("   ✓ Conclusions dan production recommendations")
        
        # Summary of best model
        results = {
            'Percobaan 1': {'rmse': 76545.71, 'time_s': 34.06, 'n_iter': 1647},
            'Percobaan 2': {'rmse': 75043.54, 'time_s': 13.99, 'n_iter': 461},
            'Percobaan 3': {'rmse': 73713.10, 'time_s': 20.13, 'n_iter': 406}
        }
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Architecture: (16, 16, 32)")
        print(f"   RMSE: {results[best_model]['rmse']:.2f}")
        print(f"   Training Time: {results[best_model]['time_s']:.2f}s")
        print(f"   Iterations: {results[best_model]['n_iter']}")
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

if __name__ == "__main__":
    main()