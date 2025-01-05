import csv
import os
import math

import pandas as pd


def check_gellerstam_restrictions(token, TE, OS, 
        te_token_rel : float, 
        os_token_rel : float, 
        rank : int = 1000, 
        more_common : float = 0.7,
        direction = 'TE'
    ):
    # Condition (1) in Gellerstam (1986)
    #if TE['all'].index.get_loc(token) > rank:
    #    return False
    if TE['all'].loc[token]['count'] < rank:
        return False
    # Condition (2) in Gellerstam (1986)
    if not all( (token in te.index and token in osv.index) for te, osv in zip(TE.values(), OS.values())):
        return False
    # Condition (3) in Gellerstam (1986)
    token_rel_sum = te_token_rel + os_token_rel
    threshold = token_rel_sum * more_common
    direction_token_rel = te_token_rel if direction == 'TE' else os_token_rel
    if direction_token_rel < threshold:
        return False
    # My own sensible condition
    if token.isnumeric() or token in ["(", ")", "/", "-", "+", "'", '"']:
        return False
    return True


def extract_translationese_vocabulary(TE, OS, **args):
    vocab = []
    TE_token_freq_sum = TE['all']['count'].sum()
    OS_token_freq_sum = OS['all']['count'].sum()
    for t in set(TE['all'].index) & set(OS['all'].index):
        te_token_freq = TE['all'].loc[t]['count']
        os_token_freq = OS['all'].loc[t]['count']
        te_token_rel = te_token_freq / TE_token_freq_sum
        os_token_rel = os_token_freq / OS_token_freq_sum
        ll = LL(te_token_freq, os_token_freq, TE_token_freq_sum, OS_token_freq_sum)
        if check_gellerstam_restrictions(t, TE, OS, te_token_rel, os_token_rel, **args) and \
            ll >= 3.84:
            entry = {
                'token': t, 
                'TE_rel_freq': te_token_rel,
                'OS_rel_freq': os_token_rel,
                'TE_freq': te_token_freq,
                'OS_freq': os_token_freq,
                'TE_freq_per_million': te_token_rel * 1000000,
                'OS_freq_per_million': os_token_rel * 1000000,
                'LL': ll

            }
            vocab.append(entry)
    return pd.DataFrame.from_records(vocab)
    

def read_old_sb_format(fp, nrows=None):
    df = pd.read_csv(
        fp,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        header=None,
        nrows=nrows,
        names=['token', 'pos', 'lemgram', '+/-', 'raw_freq', 'rel_freq'],
        usecols=['token', 'raw_freq']
        )
    df = df.groupby('token').sum('raw_freq').rename(columns={'raw_freq': 'count'})
    df = df.sort_values('count', ascending=False)
    return df


def read_sb_format(fp, nrows=None):
    df = pd.read_csv(
        fp,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        nrows=nrows,
        usecols=['token', 'count']
        )\
        .groupby('token')\
        .sum()\
        .sort_values('count', ascending=False)
    return df


def read_opus_mt_format(fp, nrows=None):
    return pd.read_csv(
        fp, 
        compression='gzip', 
        quoting=csv.QUOTE_NONE, 
        delim_whitespace=True, 
        header=None,
        nrows=nrows,
        names=['count', 'token'])\
        .set_index('token')\
        .sort_values('count', ascending=False)

def LL(wordfreq_1, wordfreq_2, totalfreq_1, totalfreq_2):
    a = wordfreq_1
    b = wordfreq_2
    c = totalfreq_1
    d = totalfreq_2
    E1 = c*(a+b) / (c+d)
    E2 = d*(a+b) / (c+d)
    G2 = 2*((a*math.log(a/E1)) + (b*math.log(b/E2)))
    return G2

def corpus_level_statistics(TE, OS):
    entries = []
    for fn, df in (TE | OS).items():
        n_tokens = df['count'].sum()
        n_types = len(df.index.unique())
        e = {
            'n_tokens': n_tokens, 
            'n_types': n_types,
            'fn': fn
        }
        entries.append(e)
    for e in entries:
        if e['fn'] in TE:
            e['collection'] = 'TE'
        else:
            e['collection'] = 'OS'
    return pd.DataFrame.from_records(entries)


if __name__ == '__main__':
    NROWS = None #500000
    freq_folder = 'data/frequencies'
    # Translated from English
    TE = {}
    # Original Swedish
    OS = {}
    for fn in os.listdir(freq_folder):
        fp = f'{freq_folder}/{fn}'
        print("Reading ", fp)
        if fn.startswith('stats_'):
            if fn.endswith('.txt'):
                df = read_old_sb_format(fp, nrows=NROWS)
            else:
                df = read_sb_format(fp, nrows=NROWS)
            OS[fn] = df
        else:
            df = read_opus_mt_format(fp, nrows=NROWS)
            TE[fn] = df
    corpus_stats_df = corpus_level_statistics(TE, OS)
    concat = pd.concat(TE.values())
    TE['all'] = concat.groupby(concat.index).sum()
    concat = pd.concat(OS.values())
    OS['all'] = concat.groupby(concat.index).sum()
    translationese_vocab = extract_translationese_vocabulary(TE, OS, rank = 100, more_common = 0.7).sort_values('LL', ascending=False)
    normalese_vocab = extract_translationese_vocabulary(TE, OS, rank = 100, more_common = 0.7, direction = 'OS').sort_values('LL', ascending=False)
    with pd.ExcelWriter('results.xlsx') as writer:
        corpus_stats_df.to_excel(writer, 'Corpus statistics')
        translationese_vocab.to_excel(writer, 'Translationese word list')
        normalese_vocab.to_excel(writer, 'Normalese word list')
        

