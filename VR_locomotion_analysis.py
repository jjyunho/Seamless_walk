import numpy as np
import pickle
import argparse
import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats
import scipy.stats as stats
import seaborn as sns

speed_list = [100, 110, 120, 130, 140, 150, 160, 170]
angle_list = [-90, -45, 0, 45, 90]

def anova_test(data):
    result = stats.levene(
        data['100'],
        data['110'],
        data['120'],
        data['130'],
        data['140'],
        data['150'],
        data['160'],
        data['170']
    )

    print(result, 'test =', result.pvalue >= 0.05)


def tukey_test(data):

    df = pd.DataFrame({'score': list(data['100'].values) +
                          list(data['110'].values) +
                          list(data['120'].values) +
                          list(data['130'].values) +
                          list(data['140'].values) +
                          list(data['150'].values) +
                          list(data['160'].values) +
                          list(data['170'].values),
                       'group': np.repeat(['100', '110', '120', '130', '140', '150', '160', '170'], repeats=2500)})

    posthoc = pairwise_tukeyhsd(df['score'], df['group'], alpha=0.05)
    fig = posthoc.plot_simultaneous()
    print(posthoc)

def anova_test2(data):
    result = stats.levene(
        data['-90'],
        data['-45'],
        data['0'],
        data['45'],
        data['90'],
    )

    print(result, 'test =', result.pvalue >= 0.05)


def tukey_test2(data):
    df = pd.DataFrame({'score': list(data['-90'].values) +
                                list(data['-45'].values) +
                                list(data['0'].values) +
                                list(data['45'].values) +
                                list(data['90'].values),
                       'group': np.repeat(['-90', '-45', '0', '45', '90'], repeats=4000)})

    posthoc = pairwise_tukeyhsd(df['score'], df['group'], alpha=0.05)
    fig = posthoc.plot_simultaneous()
    print(posthoc)



speed_dict = {'100': [],
              '110': [],
              '120': [],
              '130': [],
              '140': [],
              '150': [],
              '160': [],
              '170': [],}

angle_dict = {'-90': [],
              '-45': [],
              '0': [],
              '45': [],
              '90': [],}

def load_data(args):

    participants = os.listdir(args.data_path)
    try:
        participants.remove('.DS_Store')
    except: pass

    for participant in participants:
        filenames = []
        for filename in os.listdir(args.data_path + participant):
            if filename[-6:] == 'pickle':
                filenames.append(filename)

        for filename in filenames:
            data = pd.read_pickle(args.data_path + participant + '/' + filename)

            for i in range(len(angle_list)):
                temp = data[: ,100 + 200*i :200 + 200*i]
                speed_dict[filename[:3]].extend(list(temp[1, :]))
                angle_dict[str(-90 + 45*i)].extend(list(temp[2, :]))

    speed_df = pd.DataFrame(speed_dict)
    angle_df = pd.DataFrame(angle_dict)

    speed_df_temp = speed_df.T
    speed_df_temp.index.name = 'spm'
    speed_df_temp.reset_index(inplace=True)
    speed_df_temp = speed_df_temp.melt(id_vars='spm')

    angle_df.loc[angle_df['-90'] > 0, '-90'] -= 180
    angle_df.loc[angle_df['90'] < 0, '90'] += 180

    ## speed option
    speed_df_temp[speed_df_temp.select_dtypes(include=[np.number]).columns] *= 5


    ## Figures
    bplot = sns.lineplot(x='spm', y='value', data=speed_df_temp, ci='sd')
    bplot.set_xlabel('Guided SPM', fontsize=12, fontdict={'weight': 'bold'})
    bplot.set_ylabel('Calculated speed (m/s)', fontsize=12, fontdict={'weight': 'bold'})
    fig = bplot.get_figure()
    fig.savefig('speed.png')
    fig.clf()

    bplot = sns.boxplot(data=angle_df, width=0.5)
    bplot.set_xlabel('Guided angle (degree)', fontsize=12, fontdict={'weight': 'bold'})
    bplot.set_ylabel('Calculated angle (degree)', fontsize=12, fontdict={'weight': 'bold'})
    fig2 = bplot.get_figure()
    fig2.savefig('angle.png')

    ## Statistical test
    print("speed statistical test")
    print("*"*30)
    anova_test(speed_df)
    tukey_test(speed_df)

    print("angle statistical test")
    print("*" * 30)
    anova_test2(angle_df)
    tukey_test2(angle_df)

    ## angle diffrence calculate
    for angle in angle_list:
        angle_df[str(angle)] = abs(angle_df[str(angle)] - angle)

    for speed in speed_list:
        speed_df[str(speed)] = abs(speed_df[str(speed)] - speed)

    test = angle_df.T
    test.index.name = 'angle'
    test.reset_index(inplace=True)
    test = test.melt(id_vars='angle')

    print('error rate angle(%):\n', angle_df[angle_df > 22.5].count(axis=0).sum()/angle_df.size * 100)
    print('error rate angle(%):\n', speed_df[speed_df > 0.3].count(axis=0).sum() / angle_df.size * 100)
    print('angle diff mean:\n', angle_df.mean())
    print('angle std:\n', angle_df.std())
    print('speed mean:\n', speed_df.mean())
    print('speed std:\n', speed_df.std())
    print('overall angle diff mean:\n', test['value'].mean())
    print('overall angle diff std:\n', test['value'].std())

    print('test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--data-path", type=str, default=f"./logs/")
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument("--file_name", type=str, default="seamless_walk_speed_accuracy")
    args = parser.parse_args()
    print("code start")

    load_data(args)