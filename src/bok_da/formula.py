def translate_iv_formula(fm):
    """
    Translate "y~x1+[x2~z2a+z2b]"
    """
    import re
    #re1 = re.compile('^(.*?)~(.*?)\\|(.*)$')
    re2 = re.compile('^(.*?)~(.*)\\[(.*?)~(.*?)\\](.*)$')

    if re2.match(fm) is not None:
        lhs = re2.sub('\\1', fm).strip()
        rhs = re2.sub('\\2\\3\\5', fm).strip()
        ivs = re2.sub('\\2\\4\\5', fm).strip()
        re_strip = re.compile('(^\\+\\s*|\\s*\\+$)')
        rhs = re_strip.sub('', rhs)
        ivs = re_strip.sub('', ivs)
        return f'{lhs}~{rhs}|{ivs}'
    else:
        return fm

def _is_stata_formula(fm):
    """
    Determine if the formula is a Stata formula
    """
    # to do


def translate_stata_formula(fm):
    """
    Translate stata formula
    """
    import re
    # to do
