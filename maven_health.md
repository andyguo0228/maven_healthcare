# Maven Healthcare Challenge

This project is my submission for the [Maven Healthcare Challenge](https://mavenanalytics.io/challenges/maven-healthcare-challenge/26). My goal is to analyze the Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS) survey data and determine if it was able to improve quality of care and service to patients. 

The HCAHPS survey is required by the Centers for Medicare and Medicaid Services (CMS) for all hospitals in the United States and is used to measure patients' perspectives on hospital care. The results are made public to encourage hospitals to improve their quality of care and service to patients, and to empower patients to make informed decisions about where they receive care.

## Overview
- **Exploratory data analysis**: Initial dataset review and cleaning; identify and address data quality issues.
- **National Level Analysis**: Analyze national scores over time and highlight critical measures for improvement.
- **State Level Analysis**: Review state level data and pinpoint areas with high and low scores.
- **Response Rate Analysis**: Evaluate the impact of response rates over time.
- **Summary and Recommendations**: Summarize key findings and provide recommendations for improving patient satisfaction.

## Exploratory Data Analysis
For this project I analyzed the data using a top-down approach. I started with a broad level analysis of national data, then moved down to a more detailed evaluation of state and response data. For all datasets, I converted the `release_period` column show only the year as an integer. All column headers were also converted to lowercase for ease of use. 

- **National Data**: 5 columns, 90 rows
  - The dataset contained the measure ID and corresponding top, middle, and bottom box percentage from 2015 to 2023. 
  - The national and measures data were merged to have a description for each measure.
- **State Data**: 6 columns, 4580 rows
  - The dataset contained the measure ID and corresponding top, middle, and bottom box percentage from 2015 to 2023 for each state.
  - Missing data for Maryland 2016 was interpolated by calculating the mean of 2015 and 2017 data. 
  - The states table was merged with the dataset to include the state name and region. 
  - Filtered dataset to include only data with the measure ID 'H_RECMND' and calculated NPS.
- **Response Data**: 5 columns, 43219 rows
  - The dataset contained the response rate of patients that completed the HCAHPS survey for each hospital in each state from 2015 to 2023.
  - 5 additional states (PR, VI, GU, MP, AS) corresponding to the unincorporated US territories were removed from the dataset. 
  - Rows with missing response rates were removed from the dataset.



```python
# Import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
```


```python
# Load national results into dataframe
national_results = pd.read_csv('data_tables/national_results.csv')

# Convert release period to datetime
national_results['Release Period'] = pd.to_datetime(
    national_results['Release Period'].str.lstrip('07_')).dt.year
national_results.columns = national_results.columns.str.lower().str.replace(' ', '_')

# Load measures into dataframe
measures = pd.read_csv('data_tables/measures.csv')
measures.columns = measures.columns.str.lower().str.replace(' ', '_')

# Merge national results and measures
national_results = national_results.merge(
    measures[['measure_id', 'measure']], on='measure_id', how='left')

national_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_period</th>
      <th>measure_id</th>
      <th>bottom-box_percentage</th>
      <th>middle-box_percentage</th>
      <th>top-box_percentage</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>H_CLEAN_HSP</td>
      <td>8</td>
      <td>18</td>
      <td>74</td>
      <td>Cleanliness of Hospital Environment</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>H_COMP_1</td>
      <td>4</td>
      <td>17</td>
      <td>79</td>
      <td>Communication with Nurses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>H_COMP_2</td>
      <td>4</td>
      <td>14</td>
      <td>82</td>
      <td>Communication with Doctors</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>H_COMP_3</td>
      <td>9</td>
      <td>23</td>
      <td>68</td>
      <td>Responsiveness of Hospital Staff</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>H_COMP_5</td>
      <td>18</td>
      <td>17</td>
      <td>65</td>
      <td>Communication about Medicines</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2023</td>
      <td>H_COMP_6</td>
      <td>14</td>
      <td>0</td>
      <td>86</td>
      <td>Discharge Information</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2023</td>
      <td>H_COMP_7</td>
      <td>6</td>
      <td>43</td>
      <td>51</td>
      <td>Care Transition</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2023</td>
      <td>H_HSP_RATING</td>
      <td>9</td>
      <td>21</td>
      <td>70</td>
      <td>Overall Hospital Rating</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2023</td>
      <td>H_QUIET_HSP</td>
      <td>10</td>
      <td>28</td>
      <td>62</td>
      <td>Quietness of Hospital Environment</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2023</td>
      <td>H_RECMND</td>
      <td>6</td>
      <td>25</td>
      <td>69</td>
      <td>Willingness to Recommend the Hospital</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 6 columns</p>
</div>




```python
# Load state results into dataframe
state_results = pd.read_csv('data_tables/state_results.csv')
state_results['Release Period'] = pd.to_datetime(
    state_results['Release Period'].str.lstrip('07_')).dt.year
state_results.columns = state_results.columns.str.lower().str.replace(' ', '_')

# Interpolate missing MD 2016 data
df = state_results.copy()
md_2015 = df[(df.state == 'MD') & (df.release_period == 2015)]
md_2017 = df[(df.state == 'MD') & (df.release_period == 2017)]

# Interpolate 2016 data using average of 2015 and 2017 data
md_inter = md_2015.copy()
for col in ['bottom-box_percentage', 'middle-box_percentage', 'top-box_percentage']:
    md_inter[col] = (md_2015[col].values + md_2017[col].values) / 2
md_inter['release_period'] = md_inter['release_period'] + 1

# Concatenate interpolated data with original dataframe
state_results = pd.concat([state_results, md_inter], ignore_index=True)

# Merge with states table to get state names and region
states = pd.read_csv('data_tables/states.csv')
states.columns = states.columns.str.lower().str.replace(' ', '_')
state_results = state_results.merge(states, on='state', how='left')

state_nps = state_results[state_results.measure_id == 'H_RECMND'].copy()
state_nps['nps'] = state_nps['top-box_percentage'] - state_nps['bottom-box_percentage']

state_nps
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_period</th>
      <th>state</th>
      <th>measure_id</th>
      <th>bottom-box_percentage</th>
      <th>middle-box_percentage</th>
      <th>top-box_percentage</th>
      <th>state_name</th>
      <th>region</th>
      <th>nps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>2015</td>
      <td>AK</td>
      <td>H_RECMND</td>
      <td>7.0</td>
      <td>23.0</td>
      <td>70.0</td>
      <td>Alaska</td>
      <td>Pacific</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2015</td>
      <td>AL</td>
      <td>H_RECMND</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>Alabama</td>
      <td>East South Central</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015</td>
      <td>AR</td>
      <td>H_RECMND</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>68.0</td>
      <td>Arkansas</td>
      <td>West South Central</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2015</td>
      <td>AZ</td>
      <td>H_RECMND</td>
      <td>6.0</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>Arizona</td>
      <td>Mountain</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2015</td>
      <td>CA</td>
      <td>H_RECMND</td>
      <td>6.0</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>California</td>
      <td>Pacific</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4549</th>
      <td>2023</td>
      <td>WA</td>
      <td>H_RECMND</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>68.0</td>
      <td>Washington</td>
      <td>Pacific</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>4559</th>
      <td>2023</td>
      <td>WI</td>
      <td>H_RECMND</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>Wisconsin</td>
      <td>East North Central</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>4569</th>
      <td>2023</td>
      <td>WV</td>
      <td>H_RECMND</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>68.0</td>
      <td>West Virginia</td>
      <td>South Atlantic</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>4579</th>
      <td>2023</td>
      <td>WY</td>
      <td>H_RECMND</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>70.0</td>
      <td>Wyoming</td>
      <td>Mountain</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>4589</th>
      <td>2016</td>
      <td>MD</td>
      <td>H_RECMND</td>
      <td>6.5</td>
      <td>27.5</td>
      <td>66.0</td>
      <td>Maryland</td>
      <td>South Atlantic</td>
      <td>59.5</td>
    </tr>
  </tbody>
</table>
<p>459 rows × 9 columns</p>
</div>




```python
# load response csv into dataframe
responses = pd.read_csv('data_tables/responses.csv')
responses['Release Period'] = pd.to_datetime(
    responses['Release Period'].str.lstrip('07_')).dt.year
responses.columns = responses.columns.str.lower().str.replace(' ', '_')
responses.rename(columns={'response_rate_(%)': 'response_rate'}, inplace=True)

# Drop unincorporated territories
states = pd.read_csv('data_tables/states.csv')
states.columns = states.columns.str.lower().str.replace(' ', '_')
responses = responses[responses['state'].isin(states['state'])]

# Drop rows with missing response rates
responses = responses[responses.response_rate != 'Not Available']
responses.response_rate = responses.response_rate.astype('int64')

responses
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_period</th>
      <th>state</th>
      <th>facility_id</th>
      <th>completed_surveys</th>
      <th>response_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>AL</td>
      <td>10001</td>
      <td>300 or more</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>AL</td>
      <td>10005</td>
      <td>300 or more</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>AL</td>
      <td>10006</td>
      <td>300 or more</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>AL</td>
      <td>10007</td>
      <td>Between 100 and 299</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>AL</td>
      <td>10008</td>
      <td>Fewer than 100</td>
      <td>28</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43207</th>
      <td>2023</td>
      <td>TX</td>
      <td>670143</td>
      <td>42</td>
      <td>28</td>
    </tr>
    <tr>
      <th>43208</th>
      <td>2023</td>
      <td>TX</td>
      <td>670259</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>43209</th>
      <td>2023</td>
      <td>TX</td>
      <td>670260</td>
      <td>454</td>
      <td>14</td>
    </tr>
    <tr>
      <th>43214</th>
      <td>2023</td>
      <td>TX</td>
      <td>670300</td>
      <td>186</td>
      <td>15</td>
    </tr>
    <tr>
      <th>43215</th>
      <td>2023</td>
      <td>TX</td>
      <td>670309</td>
      <td>164</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>37403 rows × 5 columns</p>
</div>



## National Results
- To look at the trend of HCAHPS survey scores over time, I calculated the net promoter score (NPS) for each year and plotted the results on a lineplot. NPS is a metric used to measure patient experience and satisfaction based on a single question: would you recommend this hospital to your friends and family? The NPS is calculated by subtracting the percentage of detractors (bottom box) from the percentage of promoters (top box).
- The top-box percentage for each measure was also plotted on a lineplot to visualize the trend over time. 
- To look at the distribution of responses for each measure, I created a stacked bar chart with the average top, middle, and bottom box percentages for each measure.

### Findings
- The NPS increased slightly in 2017, then remained unchanged until 2021 where a sharp decline was observed. 
- **Discharge Information** had the highest average top-box percentage at 87%, while **Care Transition** had the lowest average top-box percentage at 53%.
- The **Discharge Information** measure had no middle box responses, while **Care Transition** had the highest average middle box percentage of 42%.
- **Communication about Medicines** had the highest average bottom-box percentage at 18%, while **Communication with Nurses** and **Communication with Doctors** had the lowest average bottom-box percentage at 4%.


```python
# National NPS over time
# Filter and calculate NPS
national_nps = national_results[national_results['measure_id'] == 'H_RECMND'].copy()
national_nps['nps'] = national_nps['top-box_percentage'] - national_nps['bottom-box_percentage']

# create lineplot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(national_nps,
                  x='release_period',
                  y='nps')
plt.title('National NPS Over Time')
plt.xlabel('Year')
plt.ylabel('NPS')
sns.despine()

#annotate points
for index, row in national_nps.iterrows():
    ax.text(row['release_period'], row['nps'], f"{row['nps']}")

plt.show()
```


    
![png](maven_health_files/maven_health_7_0.png)
    



```python
# Top-box percentage change over time by measure
measure_score = national_results[national_results['measure_id'].isin(['H_RECMND', 'H_HSP_RATING']) == False]

# Plot lineplot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(measure_score,
                  x='release_period',
                  y='top-box_percentage',
                  hue='measure',
                  marker='o')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Top-box Percentage Change Over Time by Measure')
plt.xlabel('Year')
plt.ylabel('Top-box Percentage')
sns.despine()
plt.show()
```


    
![png](maven_health_files/maven_health_8_0.png)
    



```python
# Average bottom, middle, top box percetage of each measure
national_percentage = national_results[national_results['measure_id'].isin(['H_RECMND', 'H_HSP_RATING']) == False].drop(
    columns=['release_period', 'measure_id']).groupby('measure').mean().reset_index()

# Average response distribution by measure
plt.figure(figsize=(10, 6))
ax = national_percentage.plot(kind='barh',
                              x='measure',
                              stacked=True)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Average Response Distribution by Measure')
sns.despine()

# Annotate the bars
for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.text(width/2 + p.get_x(), 
                p.get_y() + p.get_height()/2, 
                '{:.0f}%'.format(width), 
                ha='center', 
                va='center', 
                color='white')

plt.show()
```


    <Figure size 1000x600 with 0 Axes>



    
![png](maven_health_files/maven_health_9_1.png)
    


## State
- To look at the overall scores of each state, the average NPS was calculated and the top 10 NPS and bottom 10 NPS states were plotted on a bar chart. 
- The data was also plotted on a choropleth map to visualize the distribution of NPS across the US.

### Findings
- **South Dakota** had the highest NPS of 76.0%, while **District of Columbia** had the lowest NPS of 50.3%.
- High NPS states were primarily in the **central region**, while low NPS states were primarily in the **mid/south Atlantic region**.



```python
# Prepare data for plot; average NPS by state name
state_name_nps = state_nps.groupby(['state_name'])['nps'].mean().reset_index()

# Sort top and bottom NPS
top_nps = state_name_nps.sort_values('nps', ascending=False).head(10)
bot_nps = state_name_nps.sort_values('nps', ascending=True).head(10)

# Add a label column to differentiate top and bottom states
top_nps['label'] = 'Top NPS'
bot_nps['label'] = 'Bottom NPS'

# Combine top and bottom NPS data
combined_nps = pd.concat([top_nps, bot_nps])

# Sort the combined data for better visualization
combined_nps = combined_nps.sort_values('nps', ascending=False)

# Create the barplot
plt.figure(figsize=(10, 8))
ax = sns.barplot(data=combined_nps, y='nps', x='state_name', hue='label', palette='Blues_r')

# Annotate bars
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 9),
                textcoords='offset points')

# modify plot
ax.get_legend().set_title('')
sns.despine()
plt.title("Top and Bottom NPS by State")
plt.xlabel('State')
plt.ylabel('NPS')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()


```


    
![png](maven_health_files/maven_health_11_0.png)
    



```python
# Prepare data for plot: average state NPS
state_nps = state_nps.groupby(['state'])['nps'].mean().reset_index()

# Plot average state NPS
pio.renderers.default = "png"
trace1 = go.Choropleth(
    locations=state_nps['state'],
    z=state_nps['nps'],
    locationmode='USA-states',
    colorscale='viridis',
    colorbar_title='NPS'
)
trace2 = go.Scattergeo(
    locations=state_nps['state'],
    locationmode='USA-states',
    text=state_nps['state'],
    mode='text'
)
layout = go.Layout(
    geo=dict(
        scope='usa',
        projection=dict(
            type='albers usa'
        ),
        center=dict(
            lat=37.0902,
            lon=-95.7129
        ),
    ),
    title = 'Average Score by State',
    title_x = 0.5,
    margin = dict(t=60, b=40, l=40, r=40),
    width=1000,
    height=600
)
data = [trace1, trace2]
usmap = go.Figure(data=data, layout=layout)
usmap.show()

```


    
![png](maven_health_files/maven_health_12_0.png)
    


## Responses
- To look at the trend of response rates over time, the average response rate was calculated for each year and plotted on a bar chart.

### Findings
- Average national response rate was highest in 2015 at 30.84% and consistently decreased every year to 22.74% in 2023.


```python
# Plot average national response rate by year

national_rr = responses.groupby('release_period')[['response_rate']].mean().round(2).reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(national_rr,
                 x='release_period',
                 y='response_rate',
                 palette='Blues_d')
plt.title('Average National Response Rate by Year')
plt.xlabel('Year')
plt.ylabel('Response Rate (%)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# annoate bars
for label in ax.containers:
    ax.bar_label(label)
plt.show()
```


    
![png](maven_health_files/maven_health_14_0.png)
    


## Summary
- The NPS increased slightly in 2017, then remained unchanged until 2021 where a sharp decline was observed. 
- **Discharge Information** had the highest average top-box percentage at 87%, while **Care Transition** had the lowest average top-box percentage at 53%.
- The **Discharge Information** measure had no middle box responses, while **Care Transition** had the highest average middle box percentage of 42%.
- **Communication about Medicines** had the highest average bottom-box percentage at 18%, while **Communication with Nurses** and **Communication with Doctors** had the lowest average bottom-box percentage at 4%.
- **South Dakota** had the highest NPS of 76.0%, while **District of Columbia** had the lowest NPS of 50.3%.
- High NPS states were primarily in the **central region**, while low NPS states were primarily in the **mid/south atlantic region**.
- Average national response rate was highest in 2015 at 30.84% and consistently decreased every year to 22.74% in 2023.

## Recommendations
- **Focus on care transition**: Providing patients and their families with a detailed care plan can ensure a smooth transition after being discharged from the hospital. Consider implementing dedicated teams or individuals responsible for managing transitions who can provide clear instructions for outpatient follow up care and address any questions or concerns. 
- **Improve communication about medications**: Ensuring patients are properly educated about their medications can improve patient satisfaction as well as care transition. Implement or enhance a medication reconciliation process to ensure that patients understand the purpose, dosage, side effects, and potential interactions of their medications.
- **Regional Differences**: Focus on improving patient experience in the mid/south Atlantic region. Hospitals with high NPS scores in the central region could share best practices and identify specific issues or challenges faced by hospitals in the mid/south Atlantic region.
- **Increase response rate**: The decreasing response rate over the years indicates reduced patient engagement in providing feedback. Consider simplifying the survey process by using a digital platform such as a mobile app or online portal. 

[Click here](https://public.tableau.com/views/HCAHPSSurveyProject/HCAHPSSurveyReport?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link) for a Tableau dashboard for this project. 

Thank you [Maven Analytics](https://mavenanalytics.io/) for organizing this awesome challenge! I learned a lot working with the dataset and I look forward to participating in more challenges in the future 🙂
