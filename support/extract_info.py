import numpy as np
import pandas as pd

outfile = '../data/match_matrix.csv'

df = pd.concat([
    pd.read_excel('../data/matches/2017.xlsx'),
    pd.read_excel('../data/matches/2016.xlsx'),
    pd.read_excel('../data/matches/2015.xlsx'),
    pd.read_excel('../data/matches/2014.xlsx'),
    pd.read_excel('../data/matches/2013.xlsx')
], axis = 0)

transform_name = lambda name: name.split(' ')[0]

df['Winner'] = df['Winner'].apply(transform_name)
df['Loser'] = df['Loser'].apply(transform_name)

def compute_elapsed(player, date):
    df_prevs = df[(df['Winner'] == player) | (df['Loser'] == player)]
    df_prevs = df_prevs[df_prevs['Date'] < date]
    last = df_prevs['Date'].max()
    return np.absolute((date - last).days)

final = df[[
    'Winner',
    'Loser',
    'WRank',
    'LRank',
    'Court',
    'Surface',
    'Round',
    'Series'
]]

with open(outfile, 'w') as f:
    f.write('Winner, Loser, WRank, LRank, Court, Surface, Round, Series, WElapsed, LElapsed, WSets, LSets\n')
    for entry in df.iterrows():
        row = entry[1]
        winner = row['Winner']
        loser = row['Loser']
        wrank = row['WRank']
        lrank = row['LRank']
        court = row['Court']
        surface = row['Surface']
        rnd = row['Round']
        series = row['Series']
        wsets = row['Wsets']
        lsets = row['Lsets']
        welaps = compute_elapsed(winner, row['Date'])
        lelaps = compute_elapsed(loser, row['Date'])
        f.write('%s, %s, %f, %f, %s, %s, %s, %s, %f, %f, %f, %f\n' % (
            winner,
            loser,
            wrank,
            lrank,
            court,
            surface,
            rnd,
            series,
            welaps,
            lelaps,
            wsets,
            lsets
        ))
