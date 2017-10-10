import pandas as pd

outfile = '../data/surface_ratio.csv'

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

players = df['Winner'].unique()

surface_xt = pd.crosstab(df['Winner'], df['Surface']).groupby(['Winner']).sum()

with open(outfile, 'w') as f:
    f.write('Player, RClay, RGrass, RHard\n')
    for player in players:
        vals = surface_xt[surface_xt.index.values == player].values[0]
        total = vals.sum()
        f.write('%s, %f, %f, %f\n' % (
            player,
            vals[0]/total,
            vals[1]/total,
            vals[2]/total
        ))
