# Hotel Booking Cancellation Prediction  
### Revenue Optimization & Booking Behavior Analysis  

---

## Business Problem

High cancellation rates create revenue uncertainty and operational inefficiencies in the hospitality industry. Unexpected booking cancellations impact occupancy forecasting, staffing allocation, and pricing strategy.

This project develops a predictive model to identify hotel bookings with high cancellation probability, enabling more proactive and strategic revenue management decisions.

---

## Project Objectives

- Identify key factors influencing hotel booking cancellations  
- Perform exploratory data analysis to understand booking behavior  
- Develop supervised classification models to predict cancellations  
- Optimize model performance using recall-oriented evaluation metrics  
- Provide data-driven business recommendations  

---

## Dataset Overview

| Item | Description |
|------|------------|
| Industry | Hospitality |
| Records | ~119,000 bookings |
| Features | 30+ booking-related variables |
| Target Variable | `is_canceled` (0 = Not Canceled, 1 = Canceled) |

Main feature categories include:

- Booking details (lead time, arrival date, length of stay)  
- Customer segmentation  
- Distribution channel  
- Deposit type  
- Previous cancellation history  

---

# Model Performance & Business Impact

## 1. Model Performance Metrics

### Performance Summary

| Metric                        | Result | Business Interpretation |
|-----------------------------|--------|--------------------------|
| Recall (Cancellation = 1)   | 0.935  | Nearly all cancellations are detected, enabling proactive intervention. |
| Accuracy                    | 0.729  | 73% of total bookings are correctly classified, sufficient for operational deployment. |
| F1-Score                    | 0.718  | Balanced precision-recall trade-off for mitigation strategy. |
| F2-Score                    | 0.834  | Emphasizes recall; effective in minimizing missed cancellations. |
| Precision                   | 0.583  | 58% of high-risk predictions are true cancellations; intervention strategy must remain efficient. |
| AUC-ROC                     | 0.896  | Strong discrimination between canceled and non-canceled bookings. |
| PR-AUC                      | 0.853  | Robust performance under class imbalance conditions. |
| Confusion Matrix            | [[13694, 8856], [865, 12402]] | High recall with acceptable false positive trade-off. |

---

### Key Metric Interpretation

**Recall (0.935) & F2-Score (0.834)**  
The model successfully captures almost all cancellations. From a business perspective, this significantly reduces the risk of undetected revenue loss. The recall-oriented optimization ensures high-risk bookings are identified early.

**Accuracy (0.729) & F1-Score (0.718)**  
Overall classification performance remains stable. While false positives exist, the trade-off is acceptable because the cost of missing a cancellation is higher than additional intervention.

**Precision (0.583)**  
Not all flagged bookings will cancel. Therefore, mitigation strategies (deposit, reminder, flexible prepayment) must be efficient and customer-sensitive.

**AUC-ROC (0.896) & PR-AUC (0.853)**  
The model has strong ranking capability. A probability threshold of ≥ 0.4 is identified as an optimal operational cut-off to balance capture rate and false alarms.

---

## 2. Business Impact (Financial Simulation)

### Revenue Protection

| Metric | Without Model | With Model | Impact |
|--------|---------------|------------|--------|
| Occupancy Rate | 13.85% | 15.51% | +11.98% uplift |
| RevPAR | €23.18 | €24.98 | +7.79% uplift |

Targeted intervention on bookings with probability ≥ 0.4 reduces no-show impact and improves capacity utilization.

---

### Risk Management

| Metric | Without Model | With Model | Impact |
|--------|--------------|-----------|--------|
| Cancellation Rate | 37.04% | 29.63% | 20% reduction |

The model supports optimized overbooking strategy and reduces empty room losses due to unexpected cancellations.

---

## 3. Decision & Strategy Metrics

### Risk-Based Deposit Strategy
- Apply deposit or prepayment for high-risk bookings (probability ≥ 0.4–0.6).
- Protect revenue while keeping low-risk customers unaffected.

### Channel Optimization
- Focus promotions on low-risk channels.
- Provide cancellation protection incentives for high-risk channels.
- Improve marketing ROI and reduce channel-based cancellation impact.

### Dynamic Pricing & Segmentation
- Premium pricing for high-risk bookings during high availability.
- Discounts for low-risk bookings to maximize occupancy.
- Probability-driven allocation enhances RevPAR optimization.

---

## 4. Strategic Comparison: With vs Without Model

| Aspect | Without Model | With Model |
|--------|--------------|------------|
| Cancellation Detection | Generic intervention; many cancellations missed | Probability-based detection; Recall 0.935 |
| Intervention Strategy | Uniform deposit & reminder | Selective, risk-based deposit |
| Occupancy | 13.85% | 15.51% |
| RevPAR | €23.18 | €24.98 |
| Cancellation Rate | 37.04% | 29.63% |
| Operational Efficiency | Static overbooking | Probability-driven overbooking |
| Pricing & Marketing | Uniform strategy | Risk-based segmentation |

---

## 5. Strategic Recommendations

### A. Risk-Based Prepayment Implementation
Deploy deposit policy for bookings with cancellation probability ≥ 0.4–0.6.

### B. Dynamic Pricing Integration
Integrate cancellation probability into pricing and room allocation decisions.

### C. Channel Risk Optimization
Shift marketing budget toward low-risk channels while mitigating high-risk segments.

### D. Operational Alignment
Use predicted probability for staffing, housekeeping planning, and room allocation.

### E. Continuous Monitoring
Monitor recall, precision, revenue uplift, and cancellation reduction regularly to recalibrate the model.

---

## 6. Implementation Framework

### Workflow

1. Real-time booking → preprocessing → cancellation probability prediction  
2. Decision engine applies threshold-based intervention  
3. Revenue management adjusts pricing and overbooking  
4. Marketing segmentation aligned with risk profile  
5. Continuous KPI monitoring and model retraining  

### Deployment Options

- Batch prediction (daily operational adjustment)  
- Real-time prediction (instant deposit/prepayment decision)  
- Integration with PMS & Channel Manager  
- KPI monitoring dashboard (BI tools)

---

## Conclusion

The recall-optimized predictive model (F2-score = 0.834) enables proactive revenue protection and operational efficiency. By prioritizing cancellation detection, the hotel can reduce revenue loss, improve occupancy and RevPAR, and implement data-driven pricing and channel strategies.
