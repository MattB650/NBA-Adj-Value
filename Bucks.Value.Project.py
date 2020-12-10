#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Packages for Web Scraping
import urllib.request
from pprint import pprint
from html_table_parser import HTMLTableParser
import pandas as pd

#Scrape 19-20 Metrics from BBall Index
def url_get_contents(url):
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()

xhtml = url_get_contents('https://www.bball-index.com/2019-20-impact-metrics/').decode('utf-8')
p = HTMLTableParser()
p.feed(xhtml)
pprint(p.tables[0]) 
df = pd.DataFrame(p.tables[0], columns=['Name',
  'Team(s)',
  'GP',
  'Minutes',
  'Position',
  'Offensive Archetype',
  'Defensive Role',
  'Simple Avg Impact',
  'O-PIPM',
  'D-PIPM',
  'PIPM',
  'O-RPM',
  'D-RPM',
  'RPM',
  'O-BPM',
  'D-BPM',
  'BPM',
  'O-RAPTOR',
  'D-RAPTOR',
  'RAPTOR'])


# In[ ]:


#Cleaning 19-20 Metrics
df = df.iloc[1:]


# In[ ]:


#Only Keep Columns of Interest
df = df[['Name',
  'GP',
  'Position',
  'O-PIPM',
  'D-PIPM']]
#Drop Final Row to fix error in scraping process
df.drop(df.tail(1).index, inplace=True)


# In[ ]:


#Scrape 18-19 Metrics
xhtml = url_get_contents('https://www.bball-index.com/2018-19-impact-metrics/').decode('utf-8')
p = HTMLTableParser()
p.feed(xhtml)
pprint(p.tables[0]) 
sheet = pd.DataFrame(p.tables[0], columns=['Name',
  'Team(s)',
  'GP',
  'Minutes',
  'Position',
  'Offensive Archetype',
  'Defensive Role',
  'Simple Avg Impact',
  'O-PIPM',
  'D-PIPM',
  'PIPM',
  'O-RPM',
  'D-RPM',
  'RPM',
  'O-BPM',
  'D-BPM',
  'BPM',
  'O-RAPTOR',
  'D-RAPTOR',
  'RAPTOR'])


# In[ ]:


#Clean 18-19
sheet = sheet.iloc[1:]


# In[ ]:


#Drop Final Row to fix error in scraping process
sheet.drop(sheet.tail(1).index, inplace=True)


# In[ ]:


#Only Keep Columns of Interest
sheet = sheet[['Name',
  'GP',
  'Position',
  'O-PIPM',
  'D-PIPM']]


# In[ ]:


#Combine DataFrames
import numpy as np
combo = pd.merge(df, sheet, how='outer', left_on='Name', right_on='Name')


# In[ ]:


#Convert to Numerics and Remove Punctuation
import numpy as np
combo.replace('', np.nan, inplace=True)
combo.replace(',','', regex=True, inplace=True)
cols = combo.columns.drop(['Name', 'Position_x', 'Position_y'])
combo[cols] = combo[cols].apply(pd.to_numeric, errors='coerce')


# In[ ]:


#Combine Columns from the two seasons
combo = combo.fillna(0)
combo['Position_x'].replace(0, np.nan, inplace=True)
combo['Position_y'].replace(0, np.nan, inplace=True)
combo['GP'] = combo['GP_x'] + combo['GP_y']
combo.drop('GP_x', axis=1, inplace=True)
combo.drop('GP_y', axis=1, inplace=True)
combo['Position'] = combo['Position_x'].fillna(combo["Position_y"])
combo.drop('Position_y', axis=1, inplace=True)
combo.drop('Position_x', axis=1, inplace=True)
combo['O-PIPM'] = (combo['O-PIPM_x'] + combo['O-PIPM_y'])/2
combo['D-PIPM'] = (combo['D-PIPM_x'] + combo['D-PIPM_y']) /2
combo.drop('O-PIPM_x', axis=1, inplace=True)
combo.drop('O-PIPM_y', axis=1, inplace=True)
combo.drop('D-PIPM_x', axis=1, inplace=True)
combo.drop('D-PIPM_y', axis=1, inplace=True)


# In[ ]:


#Read in 19-20 RAPTOR data from 538
raptor20 = pd.read_csv('~/Desktop/RAPTOR.players.csv')
raptor20 = raptor20[raptor20['poss'] > 100]
raptor20 = raptor20[['player_name', 'mp', 'raptor_offense', 'raptor_defense']]
raptor20.replace("'", "", regex=True, inplace=True)
raptor20.replace(" III", "", regex=True, inplace=True)
raptor20.replace(" IV", "", regex=True, inplace=True)
raptor20.replace(" II", "", regex=True, inplace=True)


# In[ ]:


#Read in RAPTOR data from 18-19
raptor19 = pd.read_csv('~/Desktop/Raptor.players18.csv')
raptor19 = raptor19[raptor19['season'] == 2019]
raptor19 = raptor19[['player_name', 'mp', 'raptor_offense', 'raptor_defense']]
raptor19.replace(" III", "", regex=True, inplace=True)
raptor19.replace(" IV", "", regex=True, inplace=True)
raptor19.replace(" II", "", regex=True, inplace=True)
raptor19.replace("'", "", regex=True, inplace=True)


# In[ ]:


#Join the two
raptor = pd.merge(raptor20, raptor19, how='outer', left_on='player_name', right_on='player_name')


# In[ ]:


#Clean RAPTOR
raptor = raptor.fillna(0)
raptor['Name'] = raptor['player_name']
raptor.drop('player_name', axis=1, inplace=True)
raptor.replace(" Jr.", "", regex=True, inplace=True)
raptor.replace(" Sr.", "", regex=True, inplace=True)


# In[ ]:


#Create sum columns
raptor['O-RAPTOR'] = (raptor['raptor_offense_x'] + raptor['raptor_offense_y'])/2
raptor['D-RAPTOR'] = (raptor['raptor_defense_x'] + raptor['raptor_defense_y'])/2
raptor.drop('raptor_offense_x', axis=1, inplace=True)
raptor.drop('raptor_offense_y', axis=1, inplace=True)
raptor.drop('raptor_defense_x', axis=1, inplace=True)
raptor.drop('raptor_defense_y', axis=1, inplace=True)
raptor['Minutes'] = raptor['mp_x'] + raptor['mp_y']
raptor.rename(columns={'mp_y':'19Minutes', 'mp_x':'20Minutes'}, inplace=True)


# In[ ]:


#Fix some punctuation discrepancies
raptor = raptor[['Name', 'Minutes', '20Minutes', '19Minutes', 'O-RAPTOR', 'D-RAPTOR']]
raptor.replace(" III", "", regex=True, inplace=True)
raptor.replace(" IV", "", regex=True, inplace=True)
raptor.replace(" II", "", regex=True, inplace=True)


# In[ ]:


#Bring RAPTOR and PIPM together
data = pd.merge(combo, raptor, how='left', left_on = 'Name', right_on= 'Name')


# In[ ]:


#Filter out players who have not played at all in last two seasons
#Filter Out Players Who Have Not Played at Least 300 Min in Combined Last Two Seasons
#Categorize by position to find more accurate replacement levels
data = data[data['Position'].notna()]
data['Position'] = data['Position'].map({'PF': 'F', 'SF': 'F', 'PG': 'G', 'SG': 'G', 'C': 'C'})
data = data[data['Minutes'] >= 300]
data = data[data['GP'] > 0]


# In[ ]:


#Create Per 100 possesion value
data['AdjPer100Value'] = (data['O-PIPM']*(1/4)) + (data['D-PIPM']*(5/12)) + (data['O-RAPTOR']*(1/4)) + (data['D-RAPTOR']*(1/12))/3


# In[ ]:


#Create replace-level production values
x = data.groupby('Position', as_index=False)['AdjPer100Value'].mean()
#Merge with original data
data = pd.merge(data, x, how='outer', left_on='Position', right_on='Position')


# In[ ]:


#Fix column names for clarity
data['AdjPer100Value'] = data['AdjPer100Value_x']
data['PositionAvg'] = data['AdjPer100Value_y']
data.drop('AdjPer100Value_x', axis=1, inplace=True)
data.drop('AdjPer100Value_y', axis=1, inplace=True)


# In[ ]:


#Create function to find rookies with less than 1500 minutes 
#Note: did not add replacement level minutes for rookies with more than 1500
def rookie(row):
    if ((row['Minutes'] < 1500) and (row['19Minutes']) == 0):
        return 'rookie'
    return ''
data['rookie'] = data.apply(lambda row: rookie(row), axis=1)


# In[ ]:


#Separate rooks of interest and original dataset
rooks = data[data['rookie'] == 'rookie']
data = data[data['rookie' ] == '']


# In[ ]:


#For players above .3 (league avg of AdjPer100Value), fill in minutes with .3 less than their individual AdjPer100Value
#For players below .3, fill in with replacement level minutes
rooks.loc[rooks['AdjPer100Value'] < .3, 'AdjPer100Value'] = ((rooks['AdjPer100Value']*rooks['Minutes'])/1500) + ((rooks['PositionAvg']*(1500-rooks['Minutes']))/1500)
rooks.loc[rooks['AdjPer100Value'] >= .3, 'AdjPer100Value'] = ((rooks['AdjPer100Value']*rooks['Minutes'])/1500) + (((rooks['AdjPer100Value']-.3)*(1500-rooks['Minutes']))/1500)


# In[ ]:


#Fill in low minute non-rooks with same method as for rooks of interest
low = data[data['Minutes'] < 2500]
data = data[data['Minutes'] >= 2500]
low.loc[low['AdjPer100Value'] < .3, 'AdjPer100Value'] = ((low['AdjPer100Value']*low['Minutes'])/2500) + ((low['PositionAvg']*(2500-low['Minutes']))/2500)
low.loc[low['AdjPer100Value'] >= .3, 'AdjPer100Value'] = ((low['AdjPer100Value']*low['Minutes'])/2500) + (((low['AdjPer100Value']-.3)*(2500-low['Minutes']))/2500)


# In[ ]:


#Bring all three back together
data = pd.concat([data, low], axis=0)
data = pd.concat([data, rooks], axis=0)


# In[ ]:


#Bring in ages dataset from Kaggle (https://www.kaggle.com/justinas/nba-players-data)
ages = pd.read_csv('~/Desktop/all_seasons.csv')
ages.replace(" III", "", regex=True, inplace=True)
ages.replace(" IV", "", regex=True, inplace=True)
ages.replace("'", "", regex=True, inplace=True)
ages.replace(" Jr.", "", regex=True, inplace=True)


# In[ ]:


#Take oldest observation of each player to get accurate ages for each 
ages = ages[['Name', 'age']]
ages = ages.sort_values('age').drop_duplicates('Name', keep='last')


# In[ ]:


#Add one year since data is from 2019
ages['age'] = ages['age'] + 1


# In[ ]:


#Left join with data
data = pd.merge(data, ages, how='left', left_on='Name', right_on='Name')


# In[ ]:


#Clean column name for clarity
data['Age'] = data['age']
data.drop('age', axis=1, inplace=True)


# In[ ]:


#Make all players younger than 20 have age 20, and older than 33 have age 33 to lessen penalties/gains
data.loc[data['Age'] > 33, 'Age'] = 33
data.loc[data['Age'] < 20, 'Age'] = 20


# In[ ]:


#Adjust AdjPer100Value subject to age
data.loc[data['Age'] < 27.5, 'AdjPer100Value'] = ((6+data['AdjPer100Value'])*(1+((27.5-data['Age'])/30)))-6
data.loc[data['Age'] > 27.5, 'AdjPer100Value'] = ((6+data['AdjPer100Value'])*(1+((27.5-data['Age'])/60)))-6


# In[ ]:


#Final calculation of AdjPer100Value, taking difference of value minus positional avg
data['AdjPer100Value'] = data['AdjPer100Value'] - data['PositionAvg']


# In[ ]:


#Web scrape contract data from basketball-reference
xhtml = url_get_contents('https://www.basketball-reference.com/contracts/players.html').decode('utf-8')
p = HTMLTableParser()
p.feed(xhtml)
pprint(p.tables[0]) 
contracts = pd.DataFrame(p.tables[0], columns=['Rk','Player', 'TM','2019-20',
  '2020-21',
  '2021-22',
  '2022-23',
  '2023-24',
  '2024-25',
  'Signed Using',
  'Guaranteed'])


# In[ ]:


#Clean
contracts = contracts.iloc[2:]


# In[ ]:


#More cleaning
contracts = contracts[['Player', '2019-20']]
contracts = contracts[contracts['Player'] != 'Salary']
contracts = contracts[contracts['Player'] != 'Player']


# In[ ]:


#Remove punctuation
contracts.replace("'", "", regex=True, inplace=True)


# In[ ]:


#Merge back with data
data = pd.merge(data, contracts, how='left', left_on='Name', right_on='Player')


# In[ ]:


#Take columns of interest and generate top 15
data= data[['Name', 'AdjPer100Value', '2019-20']]
data.nlargest(15, 'AdjPer100Value')


# In[ ]:


#Generate top 40
fin = data.nlargest(40, 'AdjPer100Value')
fin = fin[['Name', 'AdjPer100Value', '2019-20']]
fin['2020Salary'] = fin['2019-20']
fin.drop('2019-20', axis=1, inplace=True)


# In[ ]:


#Bring in dataset with player headshots for Tableau visualization (https://github.com/erilu/web-scraping-NBA-statistics)
headshots = pd.read_csv('~/Desktop/headshots.csv')


# In[ ]:


#Columns of Interest
headshots= headshots[['Name', 'headshot']]


# In[ ]:


#Merge with top 40
fin = pd.merge(fin, headshots, how='left', on='Name')


# In[ ]:


#Bring back in first dataset to get offensive archetypes and defensive roles
xhtml = url_get_contents('https://www.bball-index.com/2019-20-impact-metrics/').decode('utf-8')
p = HTMLTableParser()
p.feed(xhtml)
pprint(p.tables[0]) 
df = pd.DataFrame(p.tables[0], columns=['Name',
  'Team(s)',
  'GP',
  'Minutes',
  'Position',
  'Offensive Archetype',
  'Defensive Role',
  'Simple Avg Impact',
  'O-PIPM',
  'D-PIPM',
  'PIPM',
  'O-RPM',
  'D-RPM',
  'RPM',
  'O-BPM',
  'D-BPM',
  'BPM',
  'O-RAPTOR',
  'D-RAPTOR',
  'RAPTOR'])


# In[ ]:


#Clean and join with top 40
roles = df.iloc[1:]
roles = roles[['Name', 'Offensive Archetype', 'Defensive Role']]
final = pd.merge(fin, roles, how='left', left_on='Name', right_on='Name')


# In[ ]:


#Building Mitchell Robinson shot chart: web scrape through NBA stats API
from nba_api.stats.endpoints import shotchartdetail
import simplejson as json

response = shotchartdetail.ShotChartDetail(team_id=0,player_id=1629011,season_nullable='2019-20',season_type_all_star='Regular Season')
content = json.loads(response.get_json())


# In[ ]:


#Set JSON as DataFrame
results = content['resultSets'][0]
headers = results['headers']
rows = results['rowSet']
mitch = pd.DataFrame(rows)
mitch.columns = headers

#Write to csv file
mitch.to_csv('~/Desktop/mitch.csv', index=False)


# In[ ]:


#Scrape in 2018-19 shots as well
response = shotchartdetail.ShotChartDetail(team_id=0,player_id=1629011,season_nullable='2018-19',season_type_all_star='Regular Season')
content = json.loads(response.get_json())


# In[ ]:


#Set JSON as DataFrame
results = content['resultSets'][0]
headers = results['headers']
rows = results['rowSet']
mitch2 = pd.DataFrame(rows)
mitch2.columns = headers

#Write to csv file
mitch2.to_csv('~/Desktop/mitch2.csv', index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#Combined and cleaned Robinson shot chart data
mitch = pd.read_csv('~/Desktop/mitch.csv')


# In[ ]:


#Set seaborn parameters
sns.set_style("white")
sns.set_color_codes()


# In[ ]:


from matplotlib.patches import Circle, Rectangle, Arc

#Function courtesy of http://savvastjortjoglou.com/nba-shot-sharts.html
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
  
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
  
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
   
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color) 
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)
        
    for element in court_elements:
        ax.add_patch(element)
        
    return ax

plt.figure(figsize=(12,11))
draw_court(outer_lines=True)
plt.xlim(-300,300)
plt.ylim(-100,500)


# In[ ]:


from matplotlib.offsetbox import OffsetImage

#Create Plot
cmap=plt.cm.YlOrRd_r 

joint_shot_chart = sns.jointplot(mitch.LOC_X, mitch.LOC_Y,
                                 kind='kde', space=0, color=cmap(0.1),
                                 cmap=cmap, n_levels=50)

joint_shot_chart.fig.set_size_inches(12,11)

ax = joint_shot_chart.ax_joint
draw_court(ax)
ax.set_xlim(-250,250)
ax.set_ylim(422.5, -47.5)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom='off', labelleft='off')
ax.set_title('Mitchell Robinson Made Shots From 2018-2020 Seasons', 
             y=1.2, fontsize=25)
ax.set_facecolor('xkcd:charcoal grey')
fig.patch.set_facecolor('xkcd:charcoal grey')

plt.show()

