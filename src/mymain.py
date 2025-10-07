import tkinter as tk
from tkinter import ttk
import numpy as np
from dataclasses import dataclass
import scipy.io as sio 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  

@dataclass
class SimOptions:
    """仿真参数类

    Attributes:
        PktLen (int): 包长度变量（字节）
        ConvCodeRate (str): 卷积码率变量
        InterleaveBits (bool): 交织深度变量
        Modulation (str): 调制方式变量
        UseTxDiversity (bool): 发射分集变量
        UseRxDiversity (int): 接收分集变量
        FreqError (int): 频率误差变量
        ChannelModel (str): 信道模型变量
        ExpDecayTrms (float): 指数衰减参数变量
        SNR (float): 信噪比变量
        UseTxPA (int): 功放变量
        UsePhaseNoise (int): 相位噪声变量
        PhaseNoiseDbcLevel (float): 相位噪声dBc变量
        PhaseNoiseCornerFreq (float): 相位噪声拐点频率变量
        PhaseNoiseFloor (float): 相位噪声底噪变量
        PacketDetection (int): 包检测变量
        TxSpectrumShape (int): 发射功率谱测试变量
        FineTimeSync (int): 精细时间同步变量
        FreqSync (int): 频率同步变量
        PilotPhaseTrack (int): 导频相位跟踪变量
        ChannelEst (int): 信道估计变量
        RxTimingOffset (int): 接收定时偏移变量
        PktsToSimulate (int): 仿真包数变量
    """

    PktLen: int # 包长度变量（字节）
    ConvCodeRate: int # 卷积码率变量
    InterleaveBits: int # 交织深度变量
    Modulation: int # 调制方式变量
    UseTxDiversity: int # 发射分集变量
    UseRxDiversity: int # 接收分集变量
    FreqError: int # 频率误差变量
    ChannelModel: int # 信道模型变量
    ExpDecayTrms: int # 指数衰减参数变量
    SNR: int # 信噪比变量
    UseTxPA: int # 功放变量
    UsePhaseNoise: int # 相位噪声变量
    PhaseNoiseDbcLevel: int # 相位噪声dBc变量
    PhaseNoiseCornerFreq: int # 相位噪声拐点频率变量
    PhaseNoiseFloor: int # 相位噪声底噪变量
    PacketDetection: int # 包检测变量
    TxSpectrumShape: int # 发射功率谱测试变量
    FineTimeSync: int # 精细时间同步变量
    FreqSync: int # 频率同步变量
    PilotPhaseTrack: int # 导频相位跟踪变量
    ChannelEst: int # 信道估计变量
    RxTimingOffset: int # 接收定时偏移变量
    PktsToSimulate: int # 仿真包数变量


# 创建GUI界面，读取用户输入的仿真参数
def ui_read_options()-> SimOptions:
    """创建GUI界面，读取用户输入的仿真参数
    
    Returns:
        用户输入的仿真参数封装在SimOptions对象中
    """
    # 创建窗口
    root = tk.Tk()
    root.title("仿真参数设置")

    # 标头
    header=tk.Frame(root).grid(row=0, column=0, columnspan=4, sticky='nswe', pady=5)
    tk.Label(header, text="基802.11a的通信系统仿真", font=("微软雅黑", 16)).grid(row=0, column=0, columnspan=4, pady=10)

    # 创建控件
    fields = [
        ("PktLen(字节)", tk.IntVar(value="100")),
        ("编码率", tk.StringVar()),
        ("交织比特", tk.BooleanVar()),
        ("调制方式", tk.StringVar()),
        ("分集发射", tk.BooleanVar()),
        ("UseRxDiversity", tk.StringVar(value="0")),
        ("FreqError", tk.StringVar(value="0")),
        ("ChannelModel", tk.StringVar(value="AWGN")),
        ("ExpDecayTrms", tk.StringVar(value="0")),
        ("信噪比（SNR）", tk.StringVar(value="20")),
        ("UseTxPA", tk.StringVar(value="0")),
        ("UsePhaseNoise", tk.StringVar(value="0")),
        ("PhaseNoiseDbcLevel", tk.StringVar(value="0")),
        ("PhaseNoiseCornerFreq", tk.StringVar(value="0")),
        ("PhaseNoiseFloor", tk.StringVar(value="0")),
        ("PacketDetection", tk.StringVar(value="1")),
        ("TxSpectrumShape", tk.StringVar(value="0")),
        ("FineTimeSync", tk.StringVar(value="1")),
        ("FreqSync", tk.StringVar(value="1")),
        ("PilotPhaseTrack", tk.StringVar(value="1")),
        ("ChannelEst", tk.StringVar(value="1")),
        ("RxTimingOffset", tk.StringVar(value="0")),
        ("PktsToSimulate", tk.StringVar(value="100"))
    ]
    
    # 构建所有控件
    build_field(root, fields)

    # 开始仿真按钮回调
    def on_start():
        """开始仿真按钮回调函数，读取用户输入的参数并关闭窗口
        """

        result["value"] = SimOptions(
            PktLen=int(field_vars["PktLen(字节)"].get()),
            ConvCodeRate=field_vars["编码率"].get(),
            InterleaveBits=field_vars["交织比特"].get(),
            Modulation=field_vars["调制方式"].get(),
            UseTxDiversity=field_vars["分集发射"].get(),
            UseRxDiversity=int(field_vars["UseRxDiversity"].get()),
            FreqError=int(field_vars["FreqError"].get()),
            ChannelModel=field_vars["ChannelModel"].get(),
            ExpDecayTrms=float(field_vars["ExpDecayTrms"].get()),
            SNR=float(field_vars["信噪比（SNR）"].get()),
            UseTxPA=int(field_vars["UseTxPA"].get()),
            UsePhaseNoise=int(field_vars["UsePhaseNoise"].get()),
            PhaseNoiseDbcLevel=float(field_vars["PhaseNoiseDbcLevel"].get()),
            PhaseNoiseCornerFreq=float(field_vars["PhaseNoiseCornerFreq"].get()),
            PhaseNoiseFloor=float(field_vars["PhaseNoiseFloor"].get()),
            PacketDetection=int(field_vars["PacketDetection"].get()),
            TxSpectrumShape=int(field_vars["TxSpectrumShape"].get()),
            FineTimeSync=int(field_vars["FineTimeSync"].get()),
            FreqSync=int(field_vars["FreqSync"].get()),
            PilotPhaseTrack=int(field_vars["PilotPhaseTrack"].get()),
            ChannelEst=int(field_vars["ChannelEst"].get()),
            RxTimingOffset=int(field_vars["RxTimingOffset"].get()),
            PktsToSimulate=int(field_vars["PktsToSimulate"].get())
        )
        root.destroy()

    # 添加开始仿真按钮
    tk.Button(root, text="仿真开始", command=on_start).grid(
        row=(len(fields) + 1) // 2+1, column=0, columnspan=4, sticky='we', pady=10
    )

    # 提取变量
    field_vars = {label: var for label, var in fields}
    result: dict[str, SimOptions] = {}

    
    
    root.mainloop()
    return result["value"]


# 构建控件
def build_field(root, fields):
    """构建输入控件
    
    Args:
        root (tk.Tk): 主窗口
        fields (list): 控件列表，包含标签和变量
    """

    # 布局控件
    num_per_col = (len(fields) + 1) // 2  # 每列显示的行数
    for i, (label, var) in enumerate(fields):
        col = 0 if i < num_per_col else 2
        row = i if i < num_per_col else i - num_per_col
        tk.Label(root, text=label).grid(row=row+1, column=col, sticky='e', padx=5, pady=2)

        # 特殊处理
        if(label=="编码率"):
            combo=ttk.Combobox(root,textvariable=var,state='readonly')
            combo['values']=('R1/2','R2/3','R3/4')
            combo.grid(row=row+1, column=col+1, padx=5, pady=2)
            combo.current(0)
        elif(label=="调制方式"):
            combo=ttk.Combobox(root,textvariable=var,state='readonly')
            combo['values']=('BPSK','QPSK','16QAM','64QAM')
            combo.grid(row=row+1, column=col+1, padx=5, pady=2)
            combo.current(0)
        elif(label=="分集发射"):
            combo=ttk.Radiobutton(root,text="是",variable=var,value=1)
            combo.grid(row=row+1, column=col+1, padx=5, pady=2, sticky='w')
            combo=ttk.Radiobutton(root,text="否",variable=var,value=0)
            combo.grid(row=row+1, column=col+1, padx=5, pady=2, sticky='e')
            var.set(0)
        elif(label=="交织比特"):
            combo=ttk.Radiobutton(root,text="是",variable=var,value=1)
            combo.grid(row=row+1, column=col+1, padx=5, pady=2, sticky='w')
            combo=ttk.Radiobutton(root,text="否",variable=var,value=0)
            combo.grid(row=row+1, column=col+1, padx=5, pady=2, sticky='e')
            var.set(1)
        else:
            tk.Entry(root, textvariable=var).grid(row=row+1, column=col+1, padx=5, pady=2)

# 常量类
class SimConst:
    """仿真常量类.

    Attributes:
        SampFreq (float): 采样频率 (Hz)。
        ConvCodeGenPoly (np.ndarray): 卷积编码生成多项式 (2x7 数组)。
        NumSubc (int): 子载波数量。
        UsedSubcIdx (np.ndarray): 有效子载波索引。
        ShortTrainingSymbols (np.ndarray): 短训练符号序列。
        LongTrainingSymbols (np.ndarray): 长训练符号序列。
        ExtraNoiseSamples (int): 额外噪声样本数。
        PilotScramble (np.ndarray): 导频加扰序列。
        NumDataSubc (int): 数据子载波数量。
        NumPilotSubc (int): 导频子载波数量。
        DataSubcIdx (np.ndarray): 数据子载波索引。
        PilotSubcIdx (np.ndarray): 导频子载波索引。
        PilotSubcPatt (np.ndarray): 导频子载波模式。
        DataSubcPatt (np.ndarray): 数据子载波模式。
        PilotSubcSymbols (np.ndarray): 导频符号序列。
    """
    
    def __init__(self):
        self.SampFreq = 20e6  # 采样频率
        self.ConvCodeGenPoly = np.array([[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]])  # 卷积编码生成多项式
        self.NumSubc = 52
        self.UsedSubcIdx = np.concatenate((np.arange(7, 33), np.arange(34, 60)))  # MATLAB索引是闭区间
        self.ShortTrainingSymbols = np.sqrt(13/6) * np.array([
            0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0,
            0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0
        ])
        self.LongTrainingSymbols = np.array([
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1
        ])
        self.ExtraNoiseSamples = 500
        self.PilotScramble = np.array([
            1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1,
            1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1,
            1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1,
            -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1
        ])
        self.NumDataSubc = 48
        self.NumPilotSubc = 4
        self.DataSubcIdx = np.concatenate((np.arange(7, 12), np.arange(13, 26), np.arange(27, 33), np.arange(34, 40), np.arange(41, 54), np.arange(55, 60)))
        self.PilotSubcIdx = np.array([12, 26, 40, 54])
        self.PilotSubcPatt = np.array([6, 20, 33, 47])
        self.DataSubcPatt = np.concatenate((np.arange(1, 6), np.arange(7, 20), np.arange(21, 27), np.arange(27, 33), np.arange(34, 47), np.arange(48, 53)))
        self.PilotSubcSymbols = np.array([1, 1, 1, -1])

# 仿真参数封装类
class Para:
    """仿真参数封装类

    Attributes:
        ui_options(SimOptions): 用户界面输入的仿真参数（SimOptions对象）
        sim_consts (SimConst): 仿真常量（SimConst对象）
    """
    
    def __init__(self, ui_options:SimOptions, sim_consts:SimConst):
        self.ui_options = ui_options
        self.sim_consts = sim_consts

def runsim(para: Para):
    """运行仿真
    Args:
        para (Para): 仿真参数封装类对象
    """

    print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n")
    packet_count = 0
    while packet_count<para.ui_options.PktsToSimulate:
        packet_count += 1
        single_packet(para)
        # 这里添加实际的仿真代码
        # 例如：simulate_packet(sim_options, sim_consts)


def single_packet(para: Para):
    """单包仿真函数
    Args:
        para (Para): 仿真参数封装类对象
    """

    # 生成发送数据
    transmitter(para)


def transmitter(para:Para):
    """发射机处理函数

    Args:
        para (Para): 仿真参数封装类对象
    """

    # 生成数据
    # data_bits = np.random.randint(0, 2, para.ui_options.PktLen)

    # 读取MAT文件
    mat_data = sio.loadmat('src/dat/inputdat.mat')
    data_bits = mat_data['inf_bits'].flatten()
    
    # 卷积编码
    coded_bits = convolutional_encode(data_bits, para)

    # 删余
    tx_bits = tx_puncture(coded_bits, para.ui_options.ConvCodeRate)

    # 生成数字位
    rdy_to_mod_bits = tx_make_int_num_ofdm_syms(tx_bits, para)

    # 交织填充比特
    rdy_to_mod_bits = tx_interleaver(rdy_to_mod_bits, para)

    # 调制
    mod_syms = tx_modulate(rdy_to_mod_bits, para.ui_options.Modulation)

    # 传输分集
    mod_syms = tx_diversity(mod_syms, para.ui_options)

    # 添加导频符号
    if para.ui_options.UseTxDiversity == 0:
        mod_ofdm_syms = tx_add_pilot_syms(mod_syms, para.ui_options)
    else:
        mod_ofdm_syms = np.zeros_like(mod_syms)
        mod_ofdm_syms[0, :] = tx_add_pilot_syms(mod_syms[0, :], para.ui_options)  # 分集第1路
        mod_ofdm_syms[1, :] = tx_add_pilot_syms(mod_syms[1, :], para.ui_options)  # 分集第2路


def convolutional_encode(data_bits, para:Para)-> np.ndarray:
    """卷积编码函数

    1. 卷积后长度 = 生成多项式长度 + 输入比特数 - 1
    2. 数据分别与多项式卷积，同时模二计算
    3. 结果按列展开后输出

    Args:
        data_bits (np.ndarray): 输入数据比特序列
        para (Para): 仿真参数封装类对象

    Returns:
        卷积编码后的比特序列
    """

    poly = para.sim_consts.ConvCodeGenPoly
    number_rows = poly.shape[0]
    number_bits = poly.shape[1] + len(data_bits) - 1
    uncoded_bits = np.zeros((number_rows, number_bits), dtype=int)
    for row in range(number_rows):
        conv_result = np.convolve(data_bits, poly[row])
        uncoded_bits[row, :] = np.mod(conv_result, 2)
    # 按列展开
    coded_bits = uncoded_bits.T.flatten()
    return coded_bits


def tx_puncture(in_bits, code_rate)-> np.ndarray:
    """发射机删余函数

    依据码率获取删余窗口大小和保留位置
    计算尾部不足一窗的残余比特数
    对主数据按窗口重排成矩阵 puncture_table，仅保留 punc_patt 指定的行得到删余主数据 tx_table。
    对尾部残余比特按同一模式截取，得到 rem_punc_bits。
    将主数据和残余数据拼接成最终删余比特 punctured_bits。

    Args:
        in_bits (np.ndarray): 输入比特序列
        code_rate (str): 码率选择 ('R1/2', 'R2/3', 'R3/4')
    Returns:
        删余后的比特序列
    """

    # 获取删余模式和窗口大小
    punc_patt, punc_patt_size = get_punc_params(code_rate)
    num_rem_bits = len(in_bits) % punc_patt_size

    # 主数据分组删余
    main_bits = in_bits[:len(in_bits)-num_rem_bits]
    puncture_table = main_bits.reshape((-1, punc_patt_size)).T
    tx_table = puncture_table[punc_patt-1, :]  # MATLAB索引从1开始，Python从0开始

    # 剩余比特删余
    rem_bits = in_bits[len(in_bits)-num_rem_bits:]
    rem_punc_patt = np.where(punc_patt <= num_rem_bits)[0]
    rem_punc_bits = rem_bits[rem_punc_patt]

    # 拼接输出
    punctured_bits = np.concatenate([tx_table.T.flatten(), rem_punc_bits])
    return punctured_bits


def get_punc_params(code_rate)-> tuple[np.ndarray, int]:
    """获取删余参数函数

    Note:
        根据码率选择，返回对应的删余位置数组和窗口大小。

    Args:
        code_rate (str): 码率选择 ('R1/2', 'R2/3', 'R3/4')

    Returns:
        删余位置数组
        窗口大小数组
    """

    if code_rate == 'R3/4':
        # R = 3/4，删余模式：[1 2 3 x x 6]，x =删余
        punc_patt = np.array([1, 2, 3, 6])# 删除后余下的位置
        punc_patt_size = 6# 一组的数量
    elif code_rate == 'R2/3':
        # % R=2/3, 删余模式：[1 2 3 x], x = 删余
        punc_patt = np.array([1, 2, 3])
        punc_patt_size = 4
    elif code_rate == 'R1/2':
        # R=1/2, 删余模式：[1 2 3 4 5 6], x = 删余 
        punc_patt = np.array([1, 2, 3, 4, 5, 6])
        punc_patt_size = 6
    else:
        raise ValueError('未定义的编码率')
    return punc_patt, punc_patt_size


def tx_make_int_num_ofdm_syms(tx_bits, para: Para)-> np.ndarray:
    """生成整数个OFDM符号的比特序列函数

    Note:
        计算需要的OFDM符号数，确保数据比特数能填满整数个OFDM符号。
        如果启用发射分集且OFDM符号数为奇数，则增加一个OFDM符号以确保符号数为偶数。
        计算需要填充的比特数，并生成随机填充比特。
        将原始比特序列与填充比特拼接，得到最终的比特序列。

    Args:
        tx_bits (np.ndarray): 输入比特序列
        para (Para): 仿真参数封装类对象

    Returns:
        填充后的比特序列
    """
    sim_consts = para.sim_consts
    sim_options = para.ui_options

    n_tx_bits = len(tx_bits)
    n_syms = sim_consts.NumDataSubc
    n_bits_per_sym = get_bits_per_symbol(sim_options.Modulation)

    # 计算需要的OFDM符号数
    n_ofdm_syms = int(np.ceil(n_tx_bits / (n_syms * n_bits_per_sym)))

    # Radon Hurwitz变换需要偶数个OFDM符号
    if sim_options.UseTxDiversity:
        if n_ofdm_syms % 2 != 0:
            n_ofdm_syms += 1

    pad_len = n_ofdm_syms * n_syms * n_bits_per_sym - n_tx_bits
    pad_bits = np.random.randint(0, 2, pad_len) # 随机生成{0,1}填充比特
    out_bits = np.concatenate([tx_bits, pad_bits])
    return out_bits


def get_bits_per_symbol(mod_order: str) -> int:
    """根据调制方式获取每个符号的比特数

    Args:
        mod_order (str): 调制方式 ('BPSK', 'QPSK', '16QAM', '64QAM')

    Returns:
        每个符号的比特数
    """

    mod_order = mod_order.strip().upper()
    if mod_order == 'BPSK':
        return 1
    elif mod_order == 'QPSK':
        return 2
    elif mod_order == '16QAM':
        return 4
    elif mod_order == '64QAM':
        return 6
    else:
        raise ValueError('未定义的调制方式')



def tx_interleaver(in_bits, para: Para)-> np.ndarray:
    """发射机交织函数

    Note:
        如果启用交织，则根据调制方式和数据子载波数计算交织深度。
        生成单个OFDM符号的交织模式，并将其扩展到整个比特序列。应用交织模式重新排列输入比特，得到交织后的比特序列。
        如果未启用交织，则直接返回输入比特。

    Args:
        in_bits (np.ndarray): 输入比特序列
        para (Para): 仿真参数封装类对象

    Returns:
        交织后的比特序列
    """

    sim_consts = para.sim_consts
    sim_options = para.ui_options

    if sim_options.InterleaveBits == 1:
        interleaver_depth = sim_consts.NumDataSubc * get_bits_per_symbol(sim_options.Modulation)
        num_symbols = len(in_bits) // interleaver_depth

        # 生成单个符号的交织模式
        single_intlvr_patt = tx_gen_intlvr_patt(interleaver_depth, sim_options)

        # 生成整个包的交织模式
        intlvr_patt = np.tile(single_intlvr_patt, num_symbols) + \
                      np.repeat(np.arange(num_symbols) * interleaver_depth, interleaver_depth)
        # 应用交织
        interleaved_bits = np.zeros_like(in_bits)
        interleaved_bits[intlvr_patt] = in_bits
    else:
        interleaved_bits = in_bits
    return interleaved_bits


def tx_gen_intlvr_patt(interleaver_depth, sim_options, sim_consts: SimConst)-> np.ndarray:
    """生成单个OFDM符号的交织模式函数

    Args:
        interleaver_depth (int): 交织深度
        sim_options (SimOptions): 用户界面输入的仿真参数（SimOptions对象）
        sim_consts (SimConst): 仿真常量（SimConst对象）

    Returns:
        交织模式索引数组
    """
    n_syms_per_ofdm_sym = sim_consts.NumDataSubc
    s = max(interleaver_depth // n_syms_per_ofdm_sym // 2, 1)
    idx = np.arange(interleaver_depth)

    # 第一次置换
    intlvr_patt = (interleaver_depth // 16) * (idx % 16) + (idx // 16)

    # 第二次置换
    perm_patt = s * (intlvr_patt // s) + \
        (intlvr_patt + interleaver_depth - (16 * intlvr_patt // interleaver_depth)) % s

    # Python索引从0开始
    return perm_patt.astype(int)

def tx_interleaver(in_bits, para: Para)-> np.ndarray:
    """发射机交织函数

    Note:
        如果启用交织，则根据调制方式和数据子载波数计算交织深度。
        生成单个OFDM符号的交织模式，并将其扩展到
        整个比特序列。应用交织模式重新排列输入比特，得到交织后的比特序列。
        如果未启用交织，则直接返回输入比特。

    Args:
        in_bits (np.ndarray): 输入比特序列
        para (Para): 仿真参数封装类对象

    Returns:
        交织后的比特序列
    """

    sim_consts = para.sim_consts
    sim_options = para.ui_options

    if sim_options.InterleaveBits == 1:
        interleaver_depth = sim_consts.NumDataSubc * get_bits_per_symbol(sim_options.Modulation)
        num_symbols = len(in_bits) // interleaver_depth

        # 生成单个符号的交织模式
        single_intlvr_patt = tx_gen_intlvr_patt(interleaver_depth, sim_options, sim_consts)

        # 为整个数据包生成交织器模式
        intlvr_patt = np.tile(single_intlvr_patt, num_symbols) + \
                        np.repeat(np.arange(num_symbols) * interleaver_depth, interleaver_depth)
        # 应用交织
        interleaved_bits = np.zeros_like(in_bits)
        interleaved_bits[intlvr_patt] = in_bits
    else:
        interleaved_bits = in_bits

def tx_gen_intlvr_patt(interleaver_depth, para: Para)-> np.ndarray:
    """生成单个OFDM符号的交织模式函数

    Args:
        interleaver_depth (int): 交织深度
        para (Para): 仿真参数封装类对象

    Returns:
        交织模式索引数组
    """

    sim_consts = para.sim_consts
    n_syms_per_ofdm_sym = sim_consts.NumDataSubc
    s = max(interleaver_depth // n_syms_per_ofdm_sym // 2, 1)
    idx = np.arange(interleaver_depth)

    # 第一次置换
    intlvr_patt = (interleaver_depth // 16) * (idx % 16) + (idx // 16)

    # 第二次置换
    perm_patt = s * (intlvr_patt // s) + \
        (intlvr_patt + interleaver_depth - (16 * intlvr_patt // interleaver_depth)) % s





























def testarray(arr,str):
    # 画出前100位图像
    plt.plot(arr[:100])
    plt.title(str)
    plt.show()


    

if __name__ == "__main__":
    options = ui_read_options()
    simconst = SimConst()
    para = Para(options, simconst)

    # 运行仿真
    runsim(para)

    print(options)





























