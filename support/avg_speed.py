'''
Support script to parse the point to point dataset and compute the average service speed for each Player

Player name saved as Surname only; for multiple surnames, only keep the last
'''

import pandas as pd

outfile = '../data/avg_service_speed.csv'

data_dir = '../data/pointbypoint/'
tournaments = [
    '2017-ausopen',
    '2016-wimbledon',
    '2016-usopen',
    '2016-frenchopen',
    '2016-ausopen',
    '2015-wimbledon',
    '2015-usopen',
    '2015-frenchopen',
    '2015-ausopen',
    '2014-wimbledon',
    '2014-usopen',
    '2014-frenchopen',
    '2014-ausopen',
    '2013-wimbledon',
    '2013-usopen',
    '2013-frenchopen',
    '2013-ausopen',
    '2012-wimbledon',
    '2012-usopen',
    '2012-frenchopen',
    '2012-ausopen',
    '2011-wimbledon',
    '2011-usopen',
    '2011-frenchopen',
    '2011-ausopen'
]

df_series = pd.concat([
    pd.read_csv(data_dir + tournament + '-points.csv') for tournament in tournaments
], axis = 0)

df_matches = pd.concat([
    pd.read_csv(data_dir + tournament + '-matches.csv') for tournament in tournaments
], axis = 0)

players = df_matches['player1'].unique()

def parse_name(full_name):
    try:
        return full_name.split(' ')[-1]
    except Exception as e:
        return '$Token$'

with open(outfile, 'w') as f:
    for player in players:
        df_p1 = df_matches[df_matches['player1'] == player]
        df_speed_p1 = df_series[df_series['match_id'].isin(df_p1['match_id'])][['PointServer', 'Speed_KMH']]
        df_speed_p1 = df_speed_p1[df_speed_p1['PointServer'] == 1]
        df_speed_p1.columns = ['Server', 'Speed']

        df_p2 = df_matches[df_matches['player2'] == player]
        df_speed_p2 = df_series[df_series['match_id'].isin(df_p2['match_id'])][['PointServer', 'Speed_KMH']]
        df_speed_p2 = df_speed_p2[df_speed_p2['PointServer'] == 2]
        df_speed_p2.columns = ['Server', 'Speed']

        df_speed = pd.concat([
            df_speed_p1,
            df_speed_p2
        ])

        df_speed = df_speed[df_speed['Speed'] != 0].drop(['Server'], axis=1)

        f.write('%s, %f\n' % (parse_name(player), df_speed['Speed'].mean()))
