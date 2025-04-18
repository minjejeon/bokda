def search(df, text, cols=None, invert = False, combine = 'or', regex = True):
    import warnings
    import pandas as pd
    import re
    full = df.columns.tolist()
    if cols==None: cols = full
    if isinstance(cols, str): cols = [cols]
    if isinstance(cols, int): cols = [cols]
    if isinstance(cols, list) and isinstance(cols[0], int):
        drop = [x for x in cols if x<0 or x>= df.shape[1]]
        if len(drop)>0:
            s = [str(x) for x in drop]
            print(f'*** Ignored column indices: {" ".join(s)}.')
        cols = [full[i] for i in cols if i>=0 and i<len(full)]

    drop = []
    for col in cols:
        if not col in full:
            cols.remove(col)
            drop.append(col)

    if len(drop) > 0: print('*** Ignored: ' + ', '.join(drop))
    if not len(cols):
        print('*** No valid column is specified. Return nil DataFrame.')
        return df[[False]*df.shape[0]]

    res = pd.DataFrame(index = df.index, dtype = 'bool')
    for col in cols:
        s = [str(x) for x in df[col]]
        if regex:
            res[col] = [bool(re.search(text,x)) for x in s]
        else:
            res[col] = [text in x for x in s]

    if invert:
        for col in res.columns.tolist():
            res[col] = -res[col]

    if combine=='or' or combine=='|':
        return df[res.any(axis=1)]
    elif combine=='and' or combine=='&':
        return df[res.all(axis=1)]
    else:
        warnings.warn('combine="'+combine+'" is not implemented. Using "or" instead.')
        return df[res.any(axis=1)]
