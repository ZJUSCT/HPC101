'''
从学在浙大上下载每个 Lab 的提交情况，其中有提交时间列。将各 Lab 的提交时间汇总到一个 CSV 文件中，表头如下：
person, Lab1, Lab2, Lab3, Lab4, Lab5
Lab 的开始时间在程序中定义
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Step 1: Read CSV
df = pd.read_csv("submissions.csv")

labs = ['Lab1', 'Lab2', 'Lab3', 'Lab4', 'Lab5']

# Convert submission times to datetime
for lab in labs:
    df[lab] = pd.to_datetime(df[lab], format='%Y.%m.%d %H:%M')

# Step 2: Melt to long format
melted = df.melt(id_vars='person', value_vars=labs, var_name='Lab', value_name='SubmissionTime')
melted.sort_values(by='SubmissionTime', inplace=True)

# Step 3: Assign Lab colors
lab_colors = {
    'Lab1': 'tab:blue',
    'Lab2': 'tab:orange',
    'Lab3': 'tab:green',
    'Lab4': 'tab:red',
    'Lab5': 'tab:purple'
}
melted['Color'] = melted['Lab'].map(lab_colors)

# Step 4: Define Lab start times
lab_start_times = {
    'Lab1': pd.Timestamp("2024-07-05 00:00"),
    'Lab2': pd.Timestamp("2024-07-15 00:00"),
    'Lab3': pd.Timestamp("2024-08-20 00:00"),
    'Lab4': pd.Timestamp("2024-08-03 00:00"),
    'Lab5': pd.Timestamp("2024-07-24 00:00")
}

# Step 5: Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot submissions
for lab in labs:
    subset = melted[melted['Lab'] == lab]
    ax.scatter(subset['SubmissionTime'], subset['person'], label=lab, color=lab_colors[lab], s=60)

# Plot Lab start lines
for lab in labs:
    start_time = lab_start_times[lab]
    ax.axvline(start_time, color=lab_colors[lab], linestyle='--', linewidth=1.2, label=f"{lab} Start")

# Style
ax.set_title("Gantt Chart of Lab Submissions with Start Times")
ax.set_xlabel("Submission Time")
ax.set_ylabel("Student")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.xticks(rotation=30)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Legend")
plt.tight_layout()

# Save to file
plt.savefig("submission_gantt.png", dpi=300)
print("✅ Gantt chart with start times saved as submission_gantt.png")
