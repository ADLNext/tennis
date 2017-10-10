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

df_ratio = pd.read_csv('../data/surface_ratio.csv', delimiter=', ')
df_service = pd.read_csv('../data/avg_service_speed.csv', delimiter=', ')

transform_name = lambda name: name.split(' ')[0]

df['Winner'] = df['Winner'].apply(transform_name)
df['Loser'] = df['Loser'].apply(transform_name)

def compute_elapsed(player, date):
    df_prevs = df[(df['Winner'] == player) | (df['Loser'] == player)]
    df_prevs = df_prevs[df_prevs['Date'] < date]
    last = df_prevs['Date'].max()
    return np.absolute((date - last).days)



with open(outfile, 'w') as f:
    f.write('Winner, Loser, WRank, LRank, Court, Surface, Round, Series, WElapsed, LElapsed, ')
    f.write('WRClay, WRGrass, WRHard, LRClay, LRGrass, LRHard, WAvgSpeed, LAvgSpeed, Wsets, Lsets\n')
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
        f.write('%s, %s, %f, %f, %s, %s, %s, %s, %f, %f, ' % (
            winner,
            loser,
            wrank,
            lrank,
            court,
            surface,
            rnd,
            series,
            welaps,
            lelaps
        ))
        try:
            f.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n' % (
                df_ratio[df_ratio['Player'] == winner]['RClay'].values[0],
                df_ratio[df_ratio['Player'] == winner]['RGrass'].values[0],
                df_ratio[df_ratio['Player'] == winner]['RHard'].values[0],
                df_ratio[df_ratio['Player'] == loser]['RClay'].values[0],
                df_ratio[df_ratio['Player'] == loser]['RGrass'].values[0],
                df_ratio[df_ratio['Player'] == loser]['RHard'].values[0],
                df_service[df_service['Player'] == winner]['AvgSpeed'].values[0],
                df_service[df_service['Player'] == loser]['AvgSpeed'].values[0],
                wsets,
                lsets
            ))
        except:
            f.write('nan, nan, nan, nan, nan, nan, nan, nan, nan, nan\n')
