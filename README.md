# EmployeeTurnoverPrediction
This is my notebook for trying to predict the Employee Turnover Rate. 

## Data Exploration
I first began by exploring the correlation between all the variables.

### Correlation
Correlation: ![corrPng](plots/corr.png)

What I found was that there was a strong correlation with the satisfaction, evaluation, and projectCount.

### Satisfaction, Evaluation, AverageMonthlyHours
![sat-eval-pCountDistribution](plots/sat-eval-pCountDistribution.png)

**Notes**
* Large spike in the employee satisfaction distribution, grouped at low satisfaction. Multimodal for more positive satisfaction
* Employee evaluation is also multimodal. Data range only covers half of the evaluation range. 
* AverageMonthlyHours is  bimodal. First one around 150 hours and second at 250 hours, which are within 1 std dev of the average. 
* Shows that employees that left were either underworked or overworked

### Salary
![salaryPng](plots/salaryDistribution.png)
### Department
![departmentPng](plots/departmentDistribution.png)
### ProjectCount Distribution
![projectCountPng](plots/projectCountDistribution.png)
### Eval, satisfaction, averageMonthlyHours KDE
![kdePng](/Users/ktl014/PycharmProjects/PersonalProjects/EmployeeTurnOverPrediction/plots/kdePlots.png)
Evaluation
* Bimodal distribution for employees that turnover. Low or high performance -> employees tend to leave
* Employees will stay between 0.6 to 0.8

Satisfaction
* Three peaks at 0.1, 0.4, and 0.85 for people that leave the company. Categorized to really low, low, and high for people who leave.
* In between those ranges are people who stayed. 

AverageMonthlyHours
* AverageMonthlyHours is  bimodal. First one around 150 hours and second at 250 hours, which are within 1 std dev of the average. 
* Shows that employees that left were either underworked or overworked

### ProjectCount vs AverageMonthlyHours
![pCountVsHours](plots/pCountvsHours.png)
* Correlation betweeen hours worked and number of projects as it increases, for people who leave.
* More obvious for people, who had a lot of project and worked a lot of hours
* People, who didn't turn over stayed right around the monthly hours average

### Satisfaction vs Evaluation 
![satVsEval](plots/satvseval.png)



