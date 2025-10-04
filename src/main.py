import tkinter as tk
import numpy as np
from tkinter import ttk

class SimOptions:
    """仿真参数类

    1. 该类的实例化需要传入多个tkinter变量，这些变量通常绑定到GUI控件上。
    2. 该类将这些tkinter变量的值转换为适当的类型并存储为类属性。

    Attributes:
        PktLen (int): 包长度变量（字节）
        ConvCodeRate (str): 编码率变量
        InterleaveBits (bool): 交织比特变量
        Modulation (str): 调制方式变量
        UseTxDiversity (bool): 分集发射变量
        UseRxDiversity (str): 分集接收变量
        FreqError (float): 频率误差变量
        ChannelModel (str): 信道模型变量
        ExpDecayTrms (float): 指数衰减时间变量
        SNR (float): 信噪比变量
        UseTxPA (bool): 使用发射功率放大器变量
        UsePhaseNoise (bool): 使用相位噪声变量
        PhaseNoiseDbcLevel (float): 相位噪声dBc水平变量
        PhaseNoiseCornerFreq (float): 相位噪声拐点频率变量
        PhaseNoiseFloor (float): 相位噪声底噪变量
        PacketDetection (bool): 包检测变量
        TxSpectrumShape (bool): 发射频谱形状变量
        FineTimeSync (bool): 精细时间同步变量
        FreqSync (bool): 频率同步变量
        PilotPhaseTrack (bool): 导频相位跟踪变量
        ChannelEst (bool): 信道估计变量
        RxTimingOffset (float): 接收机定时偏移变量
        PktsToSimulate (int): 每次运行的包数变量 

    Returns:
        None  
    
    Warns:
        None
    """
    def __init__(self, pkt_len_var, conv_code_rate_var, interleave_bits_var, modulation_var,
                 use_tx_div_var, use_rx_div_var, freq_error_var, channel_model_var,
                 exp_decay_trms_var, snr_var, use_tx_pa_var, use_phase_noise_var,
                 phase_noise_dbc_var, phase_noise_cfreq_var, phase_noise_floor_var,
                 packet_detection_var, tx_pwr_spectrum_test_var, fine_time_sync_var,
                 freq_sync_var, pilot_phase_tracking_var, channel_estimation_var,
                 rx_timing_offset_var, pkts_per_run_var):
        """初始化SimOptions类实例

        Attributes:
            PktLen (int): 包长度变量（字节）
            ConvCodeRate (str): 编码率变量
            InterleaveBits (bool): 交织比特变量
            Modulation (str): 调制方式变量
            UseTxDiversity (bool): 分集发射变量
            UseRxDiversity (str): 分集接收变量
            FreqError (float): 频率误差变量
            ChannelModel (str): 信道模型变量
            ExpDecayTrms (float): 指数衰减时间变量
            SNR (float): 信噪比变量
            UseTxPA (bool): 使用发射功率放大器变量
            UsePhaseNoise (bool): 使用相位噪声变量
            PhaseNoiseDbcLevel (float): 相位噪声dBc水平变量
            PhaseNoiseCornerFreq (float): 相位噪声拐点频率变量
            PhaseNoiseFloor (float): 相位噪声底噪变量
            PacketDetection (bool): 包检测变量
            TxSpectrumShape (bool): 发射频谱形状变量
            FineTimeSync (bool): 精细时间同步变量
            FreqSync (bool): 频率同步变量
            PilotPhaseTrack (bool): 导频相位跟踪变量
            ChannelEst (bool): 信道估计变量
            RxTimingOffset (float): 接收机定时偏移变量
            PktsToSimulate (int): 每次运行的包数变量 
        """
        self.PktLen = int(pkt_len_var.get())*8 # 字节转比特
        self.ConvCodeRate = conv_code_rate_var.get()
        self.InterleaveBits = int(interleave_bits_var.get())
        self.Modulation = modulation_var.get()
        self.UseTxDiversity = int(use_tx_div_var.get())
        self.UseRxDiversity = int(use_rx_div_var.get())
        self.FreqError = float(freq_error_var.get())
        self.ChannelModel = channel_model_var.get()
        self.ExpDecayTrms = float(exp_decay_trms_var.get())
        self.SNR = float(snr_var.get())
        self.UseTxPA = int(use_tx_pa_var.get())
        self.UsePhaseNoise = int(use_phase_noise_var.get())
        self.PhaseNoiseDbcLevel = float(phase_noise_dbc_var.get())
        self.PhaseNoiseCornerFreq = float(phase_noise_cfreq_var.get())
        self.PhaseNoiseFloor = float(phase_noise_floor_var.get())
        self.PacketDetection = int(packet_detection_var.get())
        self.TxSpectrumShape = int(tx_pwr_spectrum_test_var.get())
        self.FineTimeSync = int(fine_time_sync_var.get())
        self.FreqSync = int(freq_sync_var.get())
        self.PilotPhaseTrack = int(pilot_phase_tracking_var.get())
        self.ChannelEst = int(channel_estimation_var.get())
        self.RxTimingOffset = float(rx_timing_offset_var.get())
        self.PktsToSimulate = int(pkts_per_run_var.get())

def ui_read_options()-> SimOptions:
    """创建并显示GUI界面，读取用户输入的仿真参数

    Returns:
        SimOptions: 包含用户输入参数的SimOptions对象
    """
    # 创建窗口
    root = tk.Tk()
    root.title("仿真参数设置")

    # 标头
    header=tk.Frame(root).grid(row=0, column=0, columnspan=4, sticky='nswe', pady=5)
    tk.Label(header, text="基802.11a的通信系统仿真", font=("微软雅黑", 16)).grid(row=0, column=0, columnspan=4, pady=10)

    # 定义变量
    pkt_len_var = tk.IntVar(value="100")
    conv_code_rate_var = tk.StringVar()
    interleave_bits_var = tk.BooleanVar()
    modulation_var = tk.StringVar()
    use_tx_div_var = tk.BooleanVar()
    use_rx_div_var = tk.StringVar(value="0")
    freq_error_var = tk.StringVar(value="0")
    channel_model_var = tk.StringVar(value="AWGN")
    exp_decay_trms_var = tk.StringVar(value="0")
    snr_var = tk.StringVar(value="20")
    use_tx_pa_var = tk.StringVar(value="0")
    use_phase_noise_var = tk.StringVar(value="0")
    phase_noise_dbc_var = tk.StringVar(value="0")
    phase_noise_cfreq_var = tk.StringVar(value="0")
    phase_noise_floor_var = tk.StringVar(value="0")
    packet_detection_var = tk.StringVar(value="1")
    tx_pwr_spectrum_test_var = tk.StringVar(value="0")
    fine_time_sync_var = tk.StringVar(value="1")
    freq_sync_var = tk.StringVar(value="1")
    pilot_phase_tracking_var = tk.StringVar(value="1")
    channel_estimation_var = tk.StringVar(value="1")
    rx_timing_offset_var = tk.StringVar(value="0")
    pkts_per_run_var = tk.StringVar(value="100")
    
    # 创建控件
    fields = [
        ("PktLen(字节)", pkt_len_var),
        ("编码率", conv_code_rate_var),
        ("交织比特", interleave_bits_var),
        ("调制方式", modulation_var),
        ("分集发射", use_tx_div_var),
        ("UseRxDiversity", use_rx_div_var),
        ("FreqError", freq_error_var),
        ("ChannelModel", channel_model_var),
        ("ExpDecayTrms", exp_decay_trms_var),
        ("信噪比（SNR）", snr_var),
        ("UseTxPA", use_tx_pa_var),
        ("UsePhaseNoise", use_phase_noise_var),
        ("PhaseNoiseDbcLevel", phase_noise_dbc_var),
        ("PhaseNoiseCornerFreq", phase_noise_cfreq_var),
        ("PhaseNoiseFloor", phase_noise_floor_var),
        ("PacketDetection", packet_detection_var),
        ("TxSpectrumShape", tx_pwr_spectrum_test_var),
        ("FineTimeSync", fine_time_sync_var),
        ("FreqSync", freq_sync_var),
        ("PilotPhaseTrack", pilot_phase_tracking_var),
        ("ChannelEst", channel_estimation_var),
        ("RxTimingOffset", rx_timing_offset_var),
        ("PktsToSimulate", pkts_per_run_var)
    ]

    # 布局
    num_per_col = (len(fields) + 1) // 2  # 每列显示的行数
    for i, (label, var) in enumerate(fields):
        col = 0 if i < num_per_col else 2
        row = i if i < num_per_col else i - num_per_col
        tk.Label(root, text=label).grid(row=row+1, column=col, sticky='e', padx=5, pady=2)
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
        
        
    
    # 用于存储结果
    result={}
    def on_start():
        # 点击按钮时，更新参数并关闭窗口
        result['value'] = SimOptions(pkt_len_var, conv_code_rate_var, interleave_bits_var, modulation_var,
            use_tx_div_var, use_rx_div_var, freq_error_var, channel_model_var,
            exp_decay_trms_var, snr_var, use_tx_pa_var, use_phase_noise_var,
            phase_noise_dbc_var, phase_noise_cfreq_var, phase_noise_floor_var,
            packet_detection_var, tx_pwr_spectrum_test_var, fine_time_sync_var,
            freq_sync_var, pilot_phase_tracking_var, channel_estimation_var,
            rx_timing_offset_var, pkts_per_run_var)
        root.destroy()

    tk.Button(root, text="仿真开始", command=on_start).grid(
        row=num_per_col+1, column=0, columnspan=4, sticky='we', pady=10
    )

    root.mainloop()
    return result["value"]

class SimConst:
    """
    仿真常量类

    :param SampFreq: 采样频率 (Hz)
    :param ConvCodeGenPoly: 卷积编码生成多项式 (2x7数组)
    :param NumSubc: 子载波数量
    :param UsedSubcIdx: 有效子载波索引
    :param ShortTrainingSymbols: 短训练符号序列
    :param LongTrainingSymbols: 长训练符号序列
    :param ExtraNoiseSamples: 额外噪声样本数
    :param PilotScramble: 导频加扰序列
    :param NumDataSubc: 数据子载波数量
    :param NumPilotSubc: 导频子载波数量
    :param DataSubcIdx: 数据子载波索引
    :param PilotSubcIdx: 导频子载波索引
    :param PilotSubcPatt: 导频子载波模式
    :param DataSubcPatt: 数据子载波模式
    :param PilotSubcSymbols: 导频符号序列
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

class Para:
    """
    仿真参数封装类

    :param ui_options: 用户界面输入的仿真参数（SimOptions对象）
    :param sim_consts: 仿真常量（SimConst对象）
    """
    def __init__(self, ui_options:SimOptions, sim_consts:SimConst):
        self.ui_options = ui_options
        self.sim_consts = sim_consts


def runsim(para: Para):
    print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n")
    packet_count = 0
    while packet_count<para.ui_options.PktsToSimulate:
        packet_count += 1
        single_packet(para)
        # 这里添加实际的仿真代码
        # 例如：simulate_packet(sim_options, sim_consts)

def single_packet(para:Para):
    transmitter(para)


def transmitter(para:Para):
    # 生成数据
    data_bits = np.random.randint(0, 2, para.ui_options.PktLen)
    
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




def convolutional_encode(data_bits, para:Para):
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

def tx_puncture(in_bits, code_rate):
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

def get_punc_params(code_rate):
    """
    根据编码率返回删余模式和分组大小
    :param code_rate: 字符串，如 'R3/4'
    :return: (punc_patt, punc_patt_size)
    """
    if code_rate == 'R3/4':
        # R=3/4, Puncture pattern: [1 2 3 x x 6], x = punctured  R = 3/4，删余模式：[1 2 3 x x 6]，x =删余
        punc_patt = np.array([1, 2, 3, 6])# 删除后余下的位置
        punc_patt_size = 6# 一组的数量
    elif code_rate == 'R2/3':
        # % R=2/3, Puncture pattern: [1 2 3 x], x = punctured 
        punc_patt = np.array([1, 2, 3])
        punc_patt_size = 4
    elif code_rate == 'R1/2':
        # R=1/2, Puncture pattern: [1 2 3 4 5 6], x = punctured 
        punc_patt = np.array([1, 2, 3, 4, 5, 6])
        punc_patt_size = 6
    else:
        raise ValueError('未定义的编码率')
    return punc_patt, punc_patt_size

def tx_make_int_num_ofdm_syms(tx_bits, para: Para):
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
    # 随机生成{0,1}填充比特
    pad_bits = np.random.randint(0, 2, pad_len)
    out_bits = np.concatenate([tx_bits, pad_bits])
    return out_bits

def get_bits_per_symbol(mod_order: str) -> int:
    """
    根据调制方式返回每个符号的比特数
    :param mod_order: 调制方式字符串，如 'BPSK', 'QPSK', '16QAM', '64QAM'
    :return: 每个符号的比特数
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

def tx_interleaver(in_bits, para: Para):
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

def tx_gen_intlvr_patt(interleaver_depth, sim_options, sim_consts: SimConst):
    """
    生成交织器置换模式
    :param interleaver_depth: 每个OFDM符号的比特数
    :param sim_options: 仿真参数
    :param sim_consts: 仿真常量
    :return: 交织后的比特索引（从0开始）
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

def tx_interleaver(in_bits, para: Para):
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



def tx_gen_intlvr_patt(interleaver_depth, para: Para):
    """
    生成交织器置换模式
    :param interleaver_depth: 每个OFDM符号的比特数
    :param sim_options: 仿真参数
    :param sim_consts: 仿真常量
    :return: 交织后的比特索引（从0开始）
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

def tx_modulate(in_bits, modulation):
    bits_in = np.array(bits_in)
    modulation = modulation.strip().upper()

    if modulation == 'BPSK':
        table = np.exp(1j * np.array([0, -np.pi]))
        table = table[[1, 0]]  # Gray码映射
        mod_symbols = table[bits_in]
    elif modulation == 'QPSK':
        table = np.exp(1j * np.array([-3/4*np.pi, 3/4*np.pi, 1/4*np.pi, -1/4*np.pi]))
        table = table[[0, 1, 3, 2]]  # Gray码映射
        inp = bits_in.reshape(-1, 2)
        idx = inp[:, 0]*2 + inp[:, 1]
        mod_symbols = table[idx]
    elif modulation == '16QAM':
        table = []
        for k in [-3, -1, 1, 3]:
            for l in [-3, -1, 1, 3]:
                table.append((k + 1j*l)/np.sqrt(10))
        table = np.array(table)
        table = table[np.array(
            [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10]
        )]
        inp = bits_in.reshape(-1, 4)
        idx = inp[:, 0]*8 + inp[:, 1]*4 + inp[:, 2]*2 + inp[:, 3]
        mod_symbols = table[idx]
    elif modulation == '64QAM':
        table = []
        for k in [-7, -5, -3, -1, 1, 3, 5, 7]:
            for l in [-7, -5, -3, -1, 1, 3, 5, 7]:
                table.append((k + 1j*l)/np.sqrt(42))
        table = np.array(table)
        table = table[np.array([
            0, 1, 3, 2, 7, 6, 4, 5,
            8, 9, 11, 10, 15, 14, 12, 13,
            24, 25, 27, 26, 31, 30, 28, 29,
            16, 17, 19, 18, 23, 22, 20, 21,
            56, 57, 59, 58, 63, 62, 60, 61,
            48, 49, 51, 50, 55, 54, 52, 53,
            32, 33, 35, 34, 39, 38, 36, 37,
            40, 41, 43, 42, 47, 46, 44, 45
        ])]
        inp = bits_in.reshape(-1, 6)
        idx = inp[:, 0]*32 + inp[:, 1]*16 + inp[:, 2]*8 + inp[:, 3]*4 + inp[:, 4]*2 + inp[:, 5]
        mod_symbols = table[idx]
    else:
        raise ValueError('未定义的调制方式')
    return mod_symbols

def tx_diversity(in_syms, ui_options):
    if ui_options.UseTxDiversity == 1:
        # 传输分集
        n_syms = len(in_syms)
        if n_syms % 2 != 0:
            raise ValueError('使用传输分集时，OFDM符号数必须为偶数')
        out_syms = tx_radon_hurwitz(in_syms)
    else:
        out_syms = in_syms

def tx_radon_hurwitz(in_syms):
    n_syms = len(in_syms)
    if n_syms % 2 != 0:
        raise ValueError('输入符号数必须为偶数')
    out_syms = np.zeros((2, n_syms//2), dtype=complex)
    for k in range(n_syms//2):
        out_syms[0, k] = (in_syms[2*k] + 1j*in_syms[2*k+1]) / np.sqrt(2)
        out_syms[1, k] = (in_syms[2*k] - 1j*in_syms[2*k+1]) / np.sqrt(2)
    return out_syms.flatten()







































if __name__== "__main__":
    # starttime = time.time()
    # endtime = time.time()
    # print(f"UI界面运行时间: {endtime - starttime} 秒")
    ui_options = ui_read_options()
    sim_consts = SimConst()
    para = Para(ui_options, sim_consts)
    runsim(para)












