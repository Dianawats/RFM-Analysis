#!/usr/bin/env python3
"""
RFM Analysis for Customer Segmentation
Complete Python script for Elevvo Internship Assignment
Author: Student
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RFMAnalysis:
    def __init__(self, file_path):
        """Initialize the RFM analysis with data file path"""
        self.file_path = file_path
        self.df = None
        self.rfm_data = None
        self.segmented_data = None
        
    def load_and_clean_data(self):
        """Load and clean the retail dataset"""
        print("=" * 70)
        print("STEP 1: LOADING AND CLEANING DATA")
        print("=" * 70)
        
        try:
            # Load the dataset
            self.df = pd.read_excel(self.file_path)
            print(f"✅ Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)
            
        # Display basic info
        print(f"\n📊 Dataset Overview:")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Date range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        
        # Data cleaning
        print(f"\n🧹 Cleaning data...")
        
        # Remove rows with missing CustomerID
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['CustomerID'])
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        
        # Remove negative quantities and prices
        self.df = self.df[self.df['Quantity'] > 0]
        self.df = self.df[self.df['UnitPrice'] > 0]
        
        # Create total amount column
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        print(f"✅ Data cleaning completed:")
        print(f"   - Removed {initial_rows - len(self.df)} invalid rows")
        print(f"   - Final dataset: {len(self.df)} rows")
        print(f"   - Unique customers: {self.df['CustomerID'].nunique()}")
        
        return True
    
    def calculate_rfm_metrics(self):
        """Calculate Recency, Frequency, and Monetary metrics"""
        print(f"\n" + "=" * 70)
        print("STEP 2: CALCULATING RFM METRICS")
        print("=" * 70)
        
        # Set reference date (day after last purchase)
        reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        
        # Calculate RFM metrics for each customer
        rfm_metrics = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',                                   # Frequency
            'TotalAmount': 'sum'                                      # Monetary
        }).reset_index()
        
        rfm_metrics.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        print("📈 RFM Metrics Summary:")
        print(rfm_metrics[['Recency', 'Frequency', 'Monetary']].describe().round(2))
        
        self.rfm_data = rfm_metrics
        return rfm_metrics
    
    def assign_rfm_scores(self):
        """Assign scores to RFM metrics and create segments - FIXED VERSION"""
        print(f"\n" + "=" * 70)
        print("STEP 3: ASSIGNING RFM SCORES")
        print("=" * 70)
        
        # Use rank-based scoring to handle duplicate values
        # Recency: Lower values are better (recent customers)
        self.rfm_data['R_Score'] = pd.qcut(self.rfm_data['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
        
        # Frequency: Handle duplicates with custom approach
        # Many customers have frequency=1, so we need robust scoring
        freq_rank = self.rfm_data['Frequency'].rank(method='dense')
        freq_max = freq_rank.max()
        freq_bins = pd.cut(freq_rank, bins=4, labels=[1, 2, 3, 4])
        self.rfm_data['F_Score'] = freq_bins
        
        # Monetary: Higher values are better
        self.rfm_data['M_Score'] = pd.qcut(self.rfm_data['Monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
        
        # Convert to integers
        self.rfm_data['R_Score'] = self.rfm_data['R_Score'].astype(int)
        self.rfm_data['F_Score'] = self.rfm_data['F_Score'].astype(int)
        self.rfm_data['M_Score'] = self.rfm_data['M_Score'].astype(int)
        
        # Create combined scores
        self.rfm_data['RFM_Score'] = self.rfm_data['R_Score'] + self.rfm_data['F_Score'] + self.rfm_data['M_Score']
        self.rfm_data['RFM_Segment'] = self.rfm_data['R_Score'].astype(str) + self.rfm_data['F_Score'].astype(str) + self.rfm_data['M_Score'].astype(str)
        
        print("🎯 Score Distribution:")
        print(f"Recency Scores: {dict(self.rfm_data['R_Score'].value_counts().sort_index())}")
        print(f"Frequency Scores: {dict(self.rfm_data['F_Score'].value_counts().sort_index())}")
        print(f"Monetary Scores: {dict(self.rfm_data['M_Score'].value_counts().sort_index())}")
        print(f"RFM Total Scores: {dict(self.rfm_data['RFM_Score'].value_counts().sort_index())}")
        
        return self.rfm_data
    
    def segment_customers(self):
        """Group customers into meaningful segments"""
        print(f"\n" + "=" * 70)
        print("STEP 4: CUSTOMER SEGMENTATION")
        print("=" * 70)
        
        # Define segmentation rules based on RFM scores
        conditions = [
            (self.rfm_data['RFM_Score'] >= 10),
            (self.rfm_data['RFM_Score'] >= 8) & (self.rfm_data['RFM_Score'] < 10),
            (self.rfm_data['RFM_Score'] >= 6) & (self.rfm_data['RFM_Score'] < 8),
            (self.rfm_data['RFM_Score'] >= 4) & (self.rfm_data['RFM_Score'] < 6),
            (self.rfm_data['RFM_Score'] < 4)
        ]
        
        segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Lost Customers']
        
        self.rfm_data['Segment'] = np.select(conditions, segments, default='Needs Attention')
        self.segmented_data = self.rfm_data
        
        # Display segment distribution
        segment_stats = self.rfm_data['Segment'].value_counts()
        print("👥 Customer Segment Distribution:")
        for segment, count in segment_stats.items():
            percentage = (count / len(self.rfm_data)) * 100
            print(f"   - {segment}: {count} customers ({percentage:.1f}%)")
        
        return self.rfm_data
    
    def generate_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n" + "=" * 70)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for plots
        os.makedirs('rfm_plots', exist_ok=True)
        
        # 1. Segment Distribution Pie Chart
        plt.figure(figsize=(12, 10))
        segment_counts = self.rfm_data['Segment'].value_counts()
        colors = ['#2E8B57', '#3CB371', '#90EE90', '#FFB6C1', '#FF69B4', '#DC143C']
        
        plt.subplot(2, 2, 1)
        plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(segment_counts)], startangle=90)
        plt.title('Customer Segment Distribution', fontweight='bold')
        
        # 2. RFM Score Distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.rfm_data['RFM_Score'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('RFM Score')
        plt.ylabel('Number of Customers')
        plt.title('RFM Score Distribution', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Average Monetary by Segment
        plt.subplot(2, 2, 3)
        monetary_by_segment = self.rfm_data.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
        bars = plt.bar(monetary_by_segment.index, monetary_by_segment.values, color='lightcoral', alpha=0.7)
        plt.title('Average Spending by Segment', fontweight='bold')
        plt.xlabel('Segment')
        plt.ylabel('Average Monetary Value (£)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'£{height:.0f}', ha='center', va='bottom')
        
        # 4. Segment RFM Heatmap
        plt.subplot(2, 2, 4)
        segment_avg = self.rfm_data.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean()
        sns.heatmap(segment_avg, annot=True, cmap='YlOrRd', fmt='.2f', 
                   cbar_kws={'label': 'Average Score'}, square=True)
        plt.title('Average RFM Scores by Segment', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('rfm_plots/segment_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: rfm_plots/segment_analysis.png")
        
        # 5. RFM Distribution Box Plots
        plt.figure(figsize=(15, 5))
        
        metrics = ['Recency', 'Frequency', 'Monetary']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            if metric == 'Monetary':
                # Use log scale for monetary to handle outliers
                plot_data = self.rfm_data.copy()
                plot_data['Log_Monetary'] = np.log1p(plot_data['Monetary'])
                sns.boxplot(data=plot_data, x='Segment', y='Log_Monetary')
                plt.ylabel('Log(Monetary Value)')
                plt.title(f'Log {metric} Distribution by Segment')
            else:
                sns.boxplot(data=self.rfm_data, x='Segment', y=metric)
                plt.title(f'{metric} Distribution by Segment')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('rfm_plots/rfm_distributions.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: rfm_plots/rfm_distributions.png")
        
        # 6. Additional: Customer Value vs Recency Scatter Plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.rfm_data['Recency'], self.rfm_data['Monetary'], 
                            c=self.rfm_data['RFM_Score'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='RFM Score')
        plt.xlabel('Recency (Days since last purchase)')
        plt.ylabel('Monetary Value (£)')
        plt.title('Customer Value vs Recency', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('rfm_plots/value_vs_recency.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: rfm_plots/value_vs_recency.png")
        
        plt.close('all')
        print("✅ All visualizations generated successfully!")
    
    def generate_marketing_recommendations(self):
        """Generate targeted marketing strategies for each segment"""
        print(f"\n" + "=" * 70)
        print("STEP 6: MARKETING RECOMMENDATIONS")
        print("=" * 70)
        
        strategies = {
            'Champions': {
                'description': 'Your most valuable customers - recent, frequent, high spenders',
                'actions': [
                    'Implement VIP loyalty program with exclusive benefits',
                    'Offer early access to new products and collections',
                    'Provide personalized shopping experiences',
                    'Request testimonials and referrals with incentives',
                    'Send birthday/anniversary gifts and special offers'
                ],
                'budget': 'High - these customers generate the most revenue',
                'communication': 'Personalized, frequent communication'
            },
            'Loyal Customers': {
                'description': 'Regular customers with good spending patterns',
                'actions': [
                    'Develop tiered loyalty program with achievable rewards',
                    'Send personalized recommendations based on purchase history',
                    'Offer special discounts on frequently purchased categories',
                    'Invite to exclusive sales events and previews',
                    'Create "favorite brand" notifications for new arrivals'
                ],
                'budget': 'Medium-High - focus on retention and upselling',
                'communication': 'Regular engagement with relevant offers'
            },
            'Potential Loyalists': {
                'description': 'Recent customers showing potential for loyalty',
                'actions': [
                    'Create "welcome series" emails educating about brand value',
                    'Offer bundle deals or "complete the look" suggestions',
                    'Implement referral program with attractive rewards',
                    'Send surveys to understand their preferences',
                    'Provide educational content about product benefits'
                ],
                'budget': 'Medium - invest in conversion to loyalty',
                'communication': 'Educational and engagement-focused content'
            },
            'At Risk': {
                'description': 'Previously active customers who are becoming inactive',
                'actions': [
                    'Send "We miss you" campaigns with special reactivation offers',
                    'Conduct surveys to understand reasons for decreased activity',
                    'Offer personalized discounts (e.g., 25% off next purchase)',
                    'Highlight new products or improvements they might have missed',
                    'Create win-back campaign with strong value proposition'
                ],
                'budget': 'Medium - focused on reactivation',
                'communication': 'Win-back campaigns with strong incentives'
            },
            'Lost Customers': {
                'description': 'Customers who haven\'t purchased in a long time',
                'actions': [
                    'Send aggressive reactivation offers (40-50% discounts)',
                    'Re-introduce your brand and value proposition',
                    'Offer free shipping or no-minimum purchase deals',
                    'Create "what you\'ve missed" campaigns showcasing new features',
                    'Consider low-cost entry offers to rebuild relationship'
                ],
                'budget': 'Low-Medium - test reactivation effectiveness',
                'communication': 'Re-engagement campaigns with strong value'
            },
            'Needs Attention': {
                'description': 'Customers who don\'t fit other categories',
                'actions': [
                    'Send general promotional offers to gauge interest',
                    'Test different messaging and offer types',
                    'Monitor for any engagement signals',
                    'Consider them for general marketing campaigns',
                    'Segment further based on any emerging patterns'
                ],
                'budget': 'Low - monitor and test',
                'communication': 'General marketing communication'
            }
        }
        
        # Calculate segment statistics
        segment_stats = self.rfm_data.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        }).round(2)
        
        segment_stats.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
        total_customers = len(self.rfm_data)
        total_revenue = self.rfm_data['Monetary'].sum()
        
        print("\n📊 SEGMENT PERFORMANCE SUMMARY:")
        print("-" * 90)
        print(f"{'Segment':<20} {'Customers':<10} {'% of Total':<12} {'Avg Revenue':<12} {'Total Revenue':<15} {'% Revenue':<10}")
        print("-" * 90)
        
        for segment in strategies.keys():
            if segment in segment_stats.index:
                stats = segment_stats.loc[segment]
                customer_pct = (stats['Count'] / total_customers) * 100
                revenue_pct = (stats['Total_Revenue'] / total_revenue) * 100
                print(f"{segment:<20} {stats['Count']:<10} {customer_pct:<11.1f}% £{stats['Avg_Monetary']:<10.2f} £{stats['Total_Revenue']:<13.2f} {revenue_pct:<9.1f}%")
        
        print("-" * 90)
        print(f"{'TOTAL':<20} {total_customers:<10} {'100%':<12} {'-':<12} £{total_revenue:<13.2f} {'100%':<10}")
        
        print(f"\n🎯 DETAILED MARKETING STRATEGIES:")
        print("=" * 90)
        
        for segment, strategy in strategies.items():
            if segment in self.rfm_data['Segment'].value_counts():
                count = self.rfm_data[self.rfm_data['Segment'] == segment].shape[0]
                revenue = segment_stats.loc[segment, 'Total_Revenue'] if segment in segment_stats.index else 0
                print(f"\n🏷️  {segment.upper()} ({count} customers - £{revenue:,.2f} revenue)")
                print(f"📝 {strategy['description']}")
                print(f"💰 Budget Allocation: {strategy['budget']}")
                print(f"📞 Communication: {strategy['communication']}")
                print(f"🎯 Recommended Actions:")
                for i, action in enumerate(strategy['actions'], 1):
                    print(f"   {i}. {action}")
                print("-" * 70)
    
    def save_results(self):
        """Save the complete analysis results"""
        print(f"\n" + "=" * 70)
        print("STEP 7: SAVING RESULTS")
        print("=" * 70)
        
        # Save the main RFM data
        output_file = 'rfm_analysis_results.csv'
        self.rfm_data.to_csv(output_file, index=False)
        print(f"✅ RFM analysis results saved to: {output_file}")
        
        # Save segment summary
        segment_summary = self.rfm_data.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'std'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std', 'sum']
        }).round(2)
        
        segment_summary.columns = ['Count', 'Recency_Mean', 'Recency_Std', 
                                 'Frequency_Mean', 'Frequency_Std',
                                 'Monetary_Mean', 'Monetary_Std', 'Monetary_Total']
        
        segment_summary.to_csv('segment_summary.csv')
        print(f"✅ Segment summary saved to: segment_summary.csv")
        
        # Save top customers from each segment
        top_customers = self.rfm_data.nlargest(20, 'Monetary')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']]
        top_customers.to_csv('top_customers.csv', index=False)
        print(f"✅ Top customers saved to: top_customers.csv")
        
        # Generate executive summary
        self.generate_executive_summary()
    
    def generate_executive_summary(self):
        """Create an executive summary of the analysis"""
        print(f"\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        
        total_customers = len(self.rfm_data)
        total_revenue = self.rfm_data['Monetary'].sum()
        avg_customer_value = total_revenue / total_customers
        
        # Key metrics by segment
        champions = self.rfm_data[self.rfm_data['Segment'] == 'Champions']
        loyal = self.rfm_data[self.rfm_data['Segment'] == 'Loyal Customers']
        at_risk = self.rfm_data[self.rfm_data['Segment'] == 'At Risk']
        lost = self.rfm_data[self.rfm_data['Segment'] == 'Lost Customers']
        
        champions_revenue = champions['Monetary'].sum() if not champions.empty else 0
        loyal_revenue = loyal['Monetary'].sum() if not loyal.empty else 0
        top_segments_revenue = champions_revenue + loyal_revenue
        
        print(f"\n📈 KEY BUSINESS INSIGHTS:")
        print(f"   • Total Customers Analyzed: {total_customers:,}")
        print(f"   • Total Revenue: £{total_revenue:,.2f}")
        print(f"   • Average Customer Value: £{avg_customer_value:.2f}")
        print(f"   • Top Customers (Champions): {len(champions)} customers")
        print(f"   • Revenue from Top 2 Segments: £{top_segments_revenue:,.2f} ({top_segments_revenue/total_revenue*100:.1f}% of total)")
        print(f"   • At-Risk Customers: {len(at_risk)} customers needing immediate attention")
        print(f"   • Lost Customers: {len(lost)} customers requiring reactivation")
        
        print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
        print(f"   1. Focus on retaining Champions - they generate disproportionate revenue")
        print(f"   2. Implement targeted win-back campaigns for {len(at_risk)} At-Risk customers")
        print(f"   3. Develop loyalty programs to convert Potential Loyalists")
        print(f"   4. Allocate marketing budget based on segment value and potential")
        print(f"   5. Personalize communication strategies for each segment")
        
        print(f"\n📊 RESULTS DELIVERABLES:")
        print(f"   • rfm_analysis_results.csv - Detailed customer RFM data")
        print(f"   • segment_summary.csv - Segment statistics and performance")
        print(f"   • top_customers.csv - Top 20 highest-value customers")
        print(f"   • rfm_plots/ - Folder containing all visualization charts")
        
        print(f"\n✅ RFM ANALYSIS COMPLETED SUCCESSFULLY!")

def main():
    """Main function to run the complete RFM analysis"""
    print("🚀 RFM CUSTOMER SEGMENTATION ANALYSIS")
    print("Elevvo Internship Assignment - Customer Analytics")
    print("=" * 70)
    
    # Initialize the analysis
    file_path = "Online Retail.xlsx"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        print("Please make sure 'Online Retail.xlsx' is in the same folder as this script")
        return
    
    # Create and run analysis
    analyzer = RFMAnalysis(file_path)
    
    try:
        # Execute all steps
        analyzer.load_and_clean_data()
        analyzer.calculate_rfm_metrics()
        analyzer.assign_rfm_scores()
        analyzer.segment_customers()
        analyzer.generate_visualizations()
        analyzer.generate_marketing_recommendations()
        analyzer.save_results()
        
        print(f"\n🎉 RFM ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📁 Output Files Created:")
        print("   - rfm_analysis_results.csv")
        print("   - segment_summary.csv") 
        print("   - top_customers.csv")
        print("   - rfm_plots/ (multiple visualization files)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()