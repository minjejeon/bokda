import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
from matplotlib.font_manager import FontProperties
import warnings


def fix_lims(ax=None):
    ax = ax or plt.gca()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

def is_formula(s):
    return isinstance(s,str) and "~" in s

def _get_bar_widths(mykwargs, x, wd0=10, gap0=0):
    import pandas as pd
    wd = mykwargs.get('width', wd0) # width of bar
    gap = mykwargs.get('gap', gap0) # gap between bars
    if x.dtype=='datetime64[ns]':
        wd = pd.Timedelta(days=wd)
        gap = pd.Timedelta(days=gap)
    return wd,gap

def _convert_color(color, alpha=None, bright=None):
    '''
    This converts a single color by `alpha` and `bright`. `alpha`
    is in between 0 and 1, and the original color is `bright=1`.
    '''
    import matplotlib.colors as mcolors
    if alpha is not None:
        color_converter = mcolors.ColorConverter()
        color = color_converter.to_rgba(color, alpha=alpha)
    if bright is not None:
        import colorsys
        rgba = mcolors.to_rgba(color)
        rgb = rgba[:3]
        alp = rgba[3]
        hls = colorsys.rgb_to_hls(*rgb)
        b = bright
        lum = hls[1]
        lum_new = b*lum if b <= 1 else (1-1/b)+lum/b
        rgb = colorsys.hls_to_rgb(hls[0], lum_new, hls[2])
        color = rgb + (alp,)
    return color

def _value_label_color(args, kwargs):
    '''
    This convergs `args` and `kwargs` into the list of y vectors,
    the list of labels (None for none), and the list of colors
    (None if none). The colors shall be handled by the class method
    `Plotter.convert_color`.
    '''
    n = len(args) # number of arguments
    if n==1:
        if isinstance(args[0],list):
            # list interface
            y = args[0]
            labels = kwargs.get('label', None)
            if labels is None or isinstance(labels,str):
                labels = [labels]
        else:
            # single
            y = [args[0]]
            labels = [kwargs.get('label', None)]
    else:
        # tuples or elements
        y = []
        labels = []
        for arg in args:
            if isinstance(arg, tuple):
                # tuple
                y.append(arg[0])
                labels.append(arg[1])
            else:
                # elements
                y.append(arg)
                labels.append(None)

    if 1==len(labels) < len(y):
        if isinstance(labels, list):
            labels = labels*len(y)
        else:
            labels = [labels]*len(y)

    cols = kwargs.get('color', None)
    if cols is not None and not isinstance(cols,list):
        cols = [cols]*len(y)

    return y,labels,cols

class Plotter:
    def __init__(self, **kwargs):
        self.axes = [None,None]
        self.xmargin = kwargs.pop('xmargin',None)
        
        mydict = {
            'figsize': kwargs.get('figsize', (7,3.5)),
        }
        self.fig, self.axes[0] = plt.subplots(**mydict)
        
        
        #### 내외부망 이부분 수정 필요 ####
        #if platform.system() == 'Windows':
        #    self.font = kwargs.pop('font', ['Arial', 'Gulim']) # 외부망
        #    plt.rcParams['font.family'] = self.font # 외부망
        #elif platform.system() == 'Linux':
        #    self.font = kwargs.pop('font', ['NanumGothicCoding']) # 내부망
        #    plt.rcParams['font.family'] = self.font # 내부망
        #    mpl.rcParams.update(mpl.rcParamsDefault) ### 내부망, 임의수정
            
        ###############################
        #mpl.rcParams['axes.unicode_minus'] = False ### 내부망, 임의수정
        
        # Set title if provided
        #self.title = kwargs.get('title', None)
        #self.title_fontsize = kwargs.get('title_fontsize', 16)
        #if self.title is not None:
        #    self.fig.suptitle(self.title, fontsize=self.title_fontsize)
        
        mydict = {
            'color': kwargs.get(
                'xtick_color',
                kwargs.get('tick_color', 'lightgray')
            ),
            'labelcolor': kwargs.get(
                'xtick_labelcolor',
                kwargs.get('tick_labelcolor', 'gray')
            ),
            'direction': kwargs.get(
                'xtick_direction',
                kwargs.get('direction', 'in')
            )
        }
        self.set_xtick(**mydict)
        mydict = {
            'color': kwargs.get(
                'ytick_color',
                kwargs.get('tick_color', 'lightgray')
            ),
            'labelcolor': kwargs.get(
                'ytick_labelcolor',
                kwargs.get('tick_labelcolor', 'gray')
            ),
            'direction': kwargs.get(
                'ytick_direction',
                kwargs.get('direction', 'in')
            )
        }
        self.set_ytick(**mydict)
        mydict = {
            'edgecolor': kwargs.get('axes_edgecolor', 'lightgray'),
            'labelcolor': kwargs.get('axes_labelcolor', 'gray'),
        }
        self.set_axes(**mydict)
        if self.xmargin is not None: self.pack(x=self.xmargin)
        self.legend_ncol = kwargs.get(
            'ncol',
            kwargs.get('legend_ncol',3)
        )
        self.colors = [
            self.blue,
            self.brown,
            self.green,
            self.red,
            self.gray,
            self.skyblue,
        ]
        self.icolor = 0
        self.handles = []

    @property
    def laxis(self):
        return self.axes[0]

    @property
    def raxis(self):
        self.prepare_right_axis(self.axes[1] is None)
        return self.axes[1]

    def axis(self, k):
        if k: return self.raxis
        else: return self.laxis

    def get_color(self, alpha=None, bright=None):
        color = self.colors[self.icolor]
        color = _convert_color(color, alpha=alpha, bright=bright)
        self.step_color()
        return color

    def set_color(self, color_id):
        self.icolor = color_id % len(self.colors)

    def step_color(self, n=1):
        self.icolor += n
        self.icolor = self.icolor % len(self.colors)

    def reset_color(self):
        self.icolor = 0

    @property
    def color(self,k): return self.colors[k]

    @property
    def yellow(self): return '#fdac28'

    @property
    def bgcolor(self): return '#d8e3f5'

    @property
    def blue(self): return '#134a99'

    @property
    def brown(self): return '#c67e48'

    @property
    def green(self): return '#128c6f'

    @property
    def red(self): return '#fc3d33'

    @property
    def gray(self): return '#bcbcbc'

    @property
    def skyblue(self): return '#23a4d9'

    @property
    def black(self): return 'black'

    def convert_colors(self, cols, n, alpha=None, bright=None):
        '''
        Return list of `n` colors. If `cols` is None, `n` colors are created.
        If `cols` is a list whose length is smalle than `n`, then the rest
        colors are appended. All the colors are converged by `alpha` and
        `bright`.
        '''
        if cols is None:
            cols = []
            for j in range(n):
                cols.append(self.get_color())
        elif isinstance(cols, list):
            for j in range(n):
                if cols[j] is None:
                    cols[j] = self.get_color()
        else:
            cols = [cols]*n

        # alpha and bright
        if alpha is not None or bright is not None:
            for j in range(n):
                cols[j] = _convert_color(cols[j], alpha=alpha, bright=bright)

        return cols

    def line_one(self, at, x, *args, **kwargs):
        ans, = self.axes[at].plot(x,*args,**kwargs)
        if 'label' in kwargs: self.handles.append(ans)
        
    def plot_one(self, at, x, *args, **kwargs):
        ans, = self.axes[at].plot(x,*args,**kwargs)
        if 'label' in kwargs: self.handles.append(ans)

    def fill_between_one(self, at, x, *args, **kwargs):
        ans = self.axes[at].fill_between(x,*args,**kwargs)
        if 'label' in kwargs: self.handles.append(ans)

    def bar_one(self, at, x, *args, **kwargs):
        ans = self.axes[at].bar(x,*args,**kwargs)
        if 'label' in kwargs: self.handles.append(ans)

    def call_plot(self, cmd, x, *args, **kwargs):
        axis = kwargs.pop('axis',0)
        if axis==1:
            self.prepare_right_axis(kwargs.pop('copy_ylim', False))
        alpha = kwargs.pop('alpha',None)
        bright = kwargs.pop('bright',None)

        xlim = kwargs.pop('xlim',None)
        ylim = kwargs.pop('ylim',None)
        if xlim is not None:
            self.set_xlim(xlim,axis=axis)
        if ylim is not None:
            self.set_ylim(ylim,axis=axis)

        method = getattr(self, cmd+'_one')
        y,labels,cols = _value_label_color(args, kwargs)
        kwargs.pop('label',None) # important
        kwargs.pop('color',None) # important

        n = len(y)
        # colors
        cols = self.convert_colors(cols, n, alpha=alpha, bright=bright)

        if cmd=='bar' and n>1:
            wd,gap = _get_bar_widths(kwargs,x)
            kwargs.setdefault('width',wd)
            kwargs.pop('gap',None) # important
            d = wd + gap
            x1 = x.copy()-(n-1)*d/2
        else:
            x1 = x

        for k in range(n):
            mykwargs = kwargs.copy()
            mykwargs['color'] = cols[k]
            if labels[k] is not None:
                mykwargs['label'] = labels[k]
            method(axis, x1, y[k], **mykwargs)
            if cmd=='bar' and n>1: x1 += d

    def call_cuplot(self, cmd, x, *args, **kwargs):
        '''plot cumulative'''
        import numpy as np
        import pandas as pd

        if not cmd in ['fill_between', 'bar', 'line']:
            raise ValueError('Only bar and fill_between are supported.')

        # same as call_plot
        axis = kwargs.pop('axis',0)
        if axis==1:
            self.prepare_right_axis(kwargs.pop('copy_ylim', False))
        alpha = kwargs.pop('alpha', None)
        bright = kwargs.pop('bright',None)

        xlim = kwargs.pop('xlim',None)
        ylim = kwargs.pop('ylim',None)
        if xlim is not None:
            self.set_xlim(xlim,axis=axis)
        if ylim is not None:
            self.set_ylim(ylim,axis=axis)

        method = getattr(self, cmd+'_one')
        y,labels,cols = _value_label_color(args, kwargs)
        kwargs.pop('label',None) # important
        kwargs.pop('color',None) # important

        n = len(y)
        # cumulate
        def max0(x): return pd.Series(np.maximum(x,0))
        def min0(x): return pd.Series(np.minimum(x,0))

        # colors
        cols = self.convert_colors(cols, n, alpha=alpha, bright=bright)

        # positive
        y_bot = np.zeros(len(x))
        for y1 in y: y_bot += max0(y1)
        for j in range(n):
            y_top = y_bot
            y_inc = max0(y[j])
            if sum(y_inc) >= 0:
                y_bot = y_top - y_inc
                if cmd=='fill_between':
                    method(
                        axis, x, y_top, y_bot, color=cols[j], label=labels[j],
                        **kwargs
                    )
                else:
                    method(
                        axis, x, y_top, color=cols[j], label=labels[j],
                        **kwargs
                    )

        # negative
        y_bot = np.zeros(len(x))
        for y1 in y: y_bot += min0(y1)
        for j in range(n):
            y_top = y_bot
            y_inc = min0(y[j])
            if sum(y_inc) < 0:
                y_bot = y_top - y_inc
                if cmd=='fill_between':
                    method(axis, x, y_top, y_bot, color=cols[j], **kwargs)
                    # no label
                else:
                    method(axis, x, y_top, color=cols[j], **kwargs)
                    # no label

    def line(self, x, *args, **kwargs):
        self.call_plot('line', x, *args, **kwargs)
        
    def plot(self, x, *args, **kwargs):
        self.call_plot('plot', x, *args, **kwargs)

    def fill_between(self, x, *args, **kwargs):
        self.call_plot('fill_between', x, *args, **kwargs)

    def bar(self, x, *args, **kwargs):
        #kwargs.setdefault('width',10)
        self.call_plot('bar', x, *args, **kwargs)

    def cu_fill_between(self, x, *args, **kwargs):
        '''Cumulative areas'''
        self.call_cuplot('fill_between', x, *args, **kwargs)

    def cu_bar(self, x, *args, **kwargs):
        '''Cumulative bars'''
        #kwargs.setdefault('width',10)
        self.call_cuplot('bar', x, *args, **kwargs)

    def cu_line(self, x, *args, **kwargs):
        '''Cumulative lines'''
        self.call_cuplot('line', x, *args, **kwargs)

    def abline(self,h=None,v=None,**kwargs):
        axis = kwargs.pop('axis',0)
        if axis==1: self.prepare_right_axis(False)
        if h is not None:
            if isinstance(h, (int,float,str)):
                self.axes[axis].axhline(h,**kwargs)
            else:
                for h1 in h: self.axes[axis].axhline(h1,**kwargs)
        if v is not None:
            if isinstance(v, (int,float,str)):
                self.axes[axis].axvline(v,**kwargs)
            else:
                for v1 in v: self.axes[axis].axvline(v1,**kwargs)

    def fill(self, xy, *args, **kwargs):
        alpha = kwargs.pop('alpha', None)
        bright = kwargs.pop('bright',None)

        if alpha is not None or bright is not None:
            for v in ['color', 'edgecolor', 'facecolor']:
                x = kwargs.pop(v,None)
                if x is not None:
                    x = _convert_color(x, alpha=alpha, bright=bright)
                    kwargs[v] = x

        if alpha is not None and alpha < 1:
            if 'color' in kwargs:
                kwargs['facecolor'] = kwargs.pop('color')
                _ = kwargs.pop('edgecolor', None)
            
        if xy=='x':
            plt.axvspan(*args, **kwargs)
        elif xy=='y':
            plt.axhspan(*args, **kwargs)
        else:
            raise ValueError('Only "x" and "y" are allowed')

    def shadow_areas(self, grayinfo, **kwargs):
        axis = kwargs.pop('axis',0)
        _gray = grayinfo.copy()
        if len(grayinfo) % 2: # odd
            _gray.append(_gray[-1]+1)

        # date1 <= d <= date2
        for j in range(0, len(_gray), 2):
            self.fill('x', _gray[j], _gray[j+1], **kwargs)

    def text(self, x, y, *args, **kwargs):
        vadj = kwargs.pop('vadj', 0.)
        ha = kwargs.pop('ha', 'center')
        va = kwargs.pop('va', 'bottom')
        for xi,yi in zip(x,y):
            plt.text(xi, yi+vadj, str(yi), ha=ha, va=va, *args, **kwargs)

    def annotate(self, text, at='both', **kwargs):
        #from matplotlib import rcParams
        #rcParams['font.family'] = 'NanumGothicCoding'
        self.prepare_right_axis(self.axes[1] is None)
        if isinstance(text,str):
            text = [text, text]
        elif isinstance(text,list):
            if len(text)==1: text.append('')
        kwargs.setdefault('xytext', (0,5))
        kwargs.setdefault('va', 'bottom')
        kwargs.setdefault('xycoords', 'axes fraction')
        kwargs.setdefault('textcoords', 'offset points')
        xy = kwargs.get('xy')
        ha = kwargs.get('ha')
        if at in ['both', 'b', 'left', 'l', 'lr', 'rl']:
            if xy is None: kwargs['xy'] = (0,1)
            if ha is None: kwargs['ha'] = 'left'
            self.axes[0].annotate(text[0], **kwargs)
        if at in ['both', 'b', 'right', 'r', 'lr', 'rl']:
            if xy is None: kwargs['xy'] = (1,1)
            if ha is None: kwargs['ha'] = 'right'
            self.axes[1].annotate(text[1], **kwargs)

    def get_xlim(self, axis):
        self.axes[axis].get_xlim()

    def set_xlim(self, xlim, axis=0):
        if self.axes[axis] is None:
            self.prepare_right_axis(False)
        self.axes[axis].set_xlim(xlim)

    def get_ylim(self, axis):
        self.axes[axis].get_ylim()

    def set_ylim(self, ylim, axis=0):
        if self.axes[axis] is None:
            self.prepare_right_axis(False)
        self.axes[axis].set_ylim(ylim)

    def set_xtick(self, **kwargs):
        kwargs.setdefault('color', 'lightgray')
        kwargs.setdefault('labelcolor', 'gray')
        kwargs.setdefault('direction', 'in')
        plt.rc('xtick', **kwargs)

    def set_ytick(self, **kwargs):
        kwargs.setdefault('color', 'lightgray')
        kwargs.setdefault('labelcolor', 'gray')
        kwargs.setdefault('direction', 'in')
        plt.rc('ytick', **kwargs)

    def set_axes(self, **kwargs):
        kwargs.setdefault('edgecolor', 'lightgray')
        kwargs.setdefault('labelcolor', 'gray')
        plt.rc('axes', **kwargs)

    def _set_axis1(self, xy, k, what, multiple, thousand=None):
        import matplotlib.dates as mdates
        import matplotlib.ticker as ticker
        axis = self.axes[k]
        if thousand=='k' and what=='numeric' and multiple is None:
            multiple = 1000
        if xy=='x':
            obj = axis.xaxis
        elif xy=='y':
            obj = axis.yaxis
        else:
            raise ValueError('xy should be either "x" or "y"')
        if what=='year':
            #obj.set_major_locator(mdates.YearLocator())
            #obj.set_major_formatter(mdates.DateFormatter('%Y'))
            obj.set_major_locator(mdates.AutoDateLocator())
            obj.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        elif what=='integer':
            obj.set_major_locator(ticker.MaxNLocator(integer=True))
            obj.set_major_locator(ticker.MaxNLocator(integer=True))
        elif what=='numeric' and multiple is not None:
            obj.set_major_locator(ticker.MultipleLocator(multiple))
            if thousand is not None and thousand=='k':
                #----------------------------------------
                def format_ticks(x,pos):
                    if x==0:
                        return '0'
                    elif not x % 1000:
                        return f'{int(x/1000)}k'
                    elif x > 250:
                        return f'{x/1000}k'
                    else:
                        return str(x)
                #----------------------------------------
                obj.set_major_formatter(ticker.FuncFormatter(format_ticks))

    def set_xaxis(self, what='numeric', multiple=None, axis=0, **kwargs):
        if isinstance(axis, list):
            for j in axis: self._set_axis1('x', j, what, multiple)
        else:
            self._set_axis1('x', axis, what, multiple)

    def set_yaxis(
            self, what='numeric', multiple=None, thousand=None, axis=[0,1],
            **kwargs,
    ):
        self.prepare_right_axis(self.axes[1] is None)
        if not isinstance(axis, list): axis = [axis]
        if 'lim' in kwargs:
            lim = kwargs.pop('lim')
            for ax in axis: self.set_ylim(lim, axis=ax)
        for ax in axis: self._set_axis1('y', ax, what, multiple, thousand)

    def prepare_right_axis(self, copy_ylim:bool):
        if self.axes[1] is None:
            self.axes[1] = self.axes[0].twinx()
        if copy_ylim: self.axes[1].set_ylim(self.axes[0].get_ylim())
        self.axes[0].set_facecolor("none")

    # def legend(self, small=False, **kwargs):
    #     prop = {'family': kwargs.pop('family',self.font)}
            
    #     if 'ncol' in kwargs:
    #         ncol = kwargs['ncol']
    #     else:
    #         kwargs['ncol'] = min(self.legend_ncol, len(self.handles))
    #         ncol = kwargs['ncol']

    #     kwargs.setdefault('loc', 'best') # default: 'upper center'
    #     kwargs.setdefault('frameon', False)
    #     #loc = kwargs.get('loc')
    #     #kwargs.setdefault(
    #     #    'bbox_to_anchor',
    #     #    (0.5,1.2) if "upper" in loc else (1.1,0.5)
    #     #)

    #     #if ncol > 1 and \
    #     #   ("upper" in loc or "lower" in loc) and \
    #     #   len(self.handles) > ncol:
    #     if ncol > 1 and len(self.handles) > ncol:
    #         reOrder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
    #         labels = [h.get_label() for h in self.handles]
    #         lgd = plt.legend(
    #             reOrder(self.handles, ncol),
    #             reOrder(labels, ncol),
    #             prop=prop, **kwargs
    #         )
    #     else:
    #         lgd = plt.legend(handles=self.handles, prop=prop, **kwargs)
    #     #plt.subplots_adjust(top=.85)

    #     if small:
    #         for handle in lgd.legendHandles:
    #             s = handle.__class__.__name__
    #             if s == 'Rectangle':
    #                 handle.set_width(10)
    #             elif s == 'Line2D':
    #                 handle.set_linewidth(1.)
                    
    def legend(self, small=False, **kwargs):

        # --- 범례를 추가할 축 객체 지정 ---
        ax = self.axes[0] # 기본적으로 첫 번째 축 사용

        # --- 축에서 직접 핸들과 라벨 가져오기 ---
        handles, labels = ax.get_legend_handles_labels()

        # 범례에 표시할 내용이 없으면 종료
        if not handles:
            # print("No handles with labels found. Legend not created.") # 필요시 주석 해제
            return None

        # --- 폰트 속성 처리 ---
        # kwargs에서 fontproperties, prop, family 순서로 확인하고 적용
        legend_font_props = kwargs.pop('fontproperties', None)
        if legend_font_props is None:
            prop_kwarg = kwargs.pop('prop', None) # prop 키워드도 받을 수 있게
            if isinstance(prop_kwarg, FontProperties):
                legend_font_props = prop_kwarg # FontProperties 객체면 사용
            elif prop_kwarg is not None:
                warnings.warn("'prop' keyword argument for legend font should be a FontProperties object. Ignoring.", UserWarning)

        if 'family' in kwargs: # 비표준 'family' 키워드 처리
            family_kwarg = kwargs.pop('family') # kwargs에서 제거
            if legend_font_props is None: # 다른 폰트 설정이 없을 때만 적용
                 try:
                     legend_font_props = FontProperties(family=family_kwarg)
                 except Exception as e:
                     warnings.warn(f"Could not create FontProperties for family='{family_kwarg}': {e}. Using default rcParams font.", UserWarning)
            else: # 이미 다른 폰트 설정이 있으면 family는 무시
                 warnings.warn("Ignoring 'family' kwarg because 'prop' or 'fontproperties' was also provided.", UserWarning)
        # legend_font_props가 None이면 최종 legend 호출 시 rcParams 전역 설정 사용됨
        # --- 폰트 처리 끝 ---


        # --- ncol 및 기타 옵션 처리 (기존 로직 유지, self.handles 대신 handles 사용) ---
        if 'ncol' in kwargs:
            ncol = kwargs['ncol']
        else:
            # self.legend_ncol 속성이 없다면 기본값 1 사용
            default_ncol = getattr(self, 'legend_ncol', 1)
            kwargs['ncol'] = min(default_ncol, len(handles)) # self.handles 대신 handles 사용
            ncol = kwargs['ncol']
        ncol = max(1, int(ncol)) # 최소 1 보장
        kwargs['ncol'] = ncol # 최종 ncol 값 kwargs에 반영

        kwargs.setdefault('loc', 'best')
        kwargs.setdefault('frameon', False)
        # --- 기타 옵션 처리 끝 ---


        # --- 핸들/라벨 순서 재정렬 (기존 로직 유지, self.handles 대신 handles/labels 사용) ---
        if ncol > 1 and len(handles) > ncol: # self.handles 대신 handles 사용
            try:
                reOrder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
                final_handles = reOrder(handles, ncol) # self.handles 대신 handles 사용
                final_labels = reOrder(labels, ncol)   # 직접 얻은 labels 사용
            except Exception as e:
                warnings.warn(f"Could not reorder legend items: {e}. Using original order.", UserWarning)
                final_handles = handles
                final_labels = labels
        else:
            final_handles = handles
            final_labels = labels
        # --- 재정렬 끝 ---


        # --- 범례 생성 (ax.legend 호출 한 번으로!) ---
        # plt.legend() 호출 및 중복 ax.legend() 호출 제거
        lgd = None
        try:
            # fontproperties 인자로 처리된 폰트 속성 전달
            lgd = ax.legend(handles=final_handles, labels=final_labels, prop=legend_font_props, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to create legend: {e}", UserWarning)
            return None
        # --- 범례 생성 끝 ---


        # --- 'small' 옵션 처리 (기존 로직 유지) ---
        if small and lgd:
             try:
                 # matplotlib 버전에 따라 legendHandles 또는 legend_handles 시도
                 handles_to_adjust = getattr(lgd, 'legendHandles', getattr(lgd, 'legend_handles', []))
                 for handle in handles_to_adjust:
                     s = handle.__class__.__name__
                     if s == 'Rectangle':
                         try: handle.set_width(10)
                         except: pass
                     elif s == 'Line2D':
                         try: handle.set_linewidth(1.)
                         except: pass
             except Exception as e:
                 warnings.warn(f"Could not resize legend handles: {e}", UserWarning)
        # --- 'small' 옵션 처리 끝 ---

        return lgd # 최종 생성된 범례 객체 반환

    def to_front(self, axis=0):
        if axis==0:
            self.axes[0].set_zorder(self.axes[1].get_zorder()+1)
        elif axis==1:
            self.axes[1].set_zorder(self.axes[0].get_zorder()+1)

    def pack(self,*args,**kwargs):
        kwargs.setdefault('x',0)
        for ax in self.axes:
            if ax is not None: ax.margins(*args, **kwargs)
            
    def set_title(self, title, fontsize=15):
        self.fig.suptitle(title, fontsize=fontsize)

    def show(self):
        self.reset_color()
        plt.show()

    def export(self, filename, dpi=300, bbox_inches='tight'):
        '''Export file'''
        self.reset_color()
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
